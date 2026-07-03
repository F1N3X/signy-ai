"""
Téléchargeur de vidéos LSF depuis le dictionnaire Élix.
"""

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, TransferSpeedColumn,
)
from rich.console import Group

# ── Configuration ─────────────────────────────────────────────────────────────

DICT_FILE   = "lsf_dataset/metadata/elix_full_dictionary.json"
OUTPUT_DIR  = Path("lsf_dataset/videos")
MAX_WORKERS = 8
HEADERS     = {"User-Agent": "Mozilla/5.0"}

API_WORD    = "https://dico.elix-lsf.fr/dictionnaire/{word}"
API_MEANING = "https://api.elix-lsf.fr/words/{word}/meanings/{wid}"

# ── Session HTTP (une par thread) ─────────────────────────────────────────────

_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1,
            # 500 retiré volontairement : le serveur Élix retourne 500
            # sur des URLs invalides (mot avec espaces, accents mal encodés),
            # et les retrier ne fait qu'aggraver les choses.
            status_forcelist=[429, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=MAX_WORKERS,
            pool_maxsize=MAX_WORKERS,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(HEADERS)
        _thread_local.session = session
    return _thread_local.session


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """Supprime les caractères interdits dans un nom de fichier."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def get_word_ids(word: str) -> list[str]:
    """Récupère tous les IDs associés à un mot.

    Les mots peuvent contenir des espaces ou des accents (ex: « à barbe »).
    `quote(..., safe='')` encode tout sauf les caractères déjà sûrs,
    ce qui évite que l'espace devienne un segment de chemin supplémentaire
    et que le serveur réponde 500.
    """
    encoded = quote(word, safe="")
    url = API_WORD.format(word=encoded)
    try:
        response = get_session().get(url, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as exc:
        # 404 = mot absent du dico, on skippe silencieusement
        if exc.response is not None and exc.response.status_code == 404:
            return []
        raise
    return sorted(set(re.findall(r'"word_id"\s*:\s*(\d+)', response.text)))


def get_video_urls(word: str, word_id: str) -> list[str]:
    """Récupère les URLs des vidéos pour un mot et un ID donnés."""
    encoded = quote(word, safe="")
    url = API_MEANING.format(word=encoded, wid=word_id)
    response = get_session().get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    urls = []
    for key in ("wordSigns"):
        for sign in data.get(key, []):
            uri = sign.get("uri")
            if uri:
                urls.append(uri)
    return urls


# ── Dashboard ─────────────────────────────────────────────────────────────────

@dataclass
class Dashboard:
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    current_word:  str = "-"
    current_file:  str = "-"
    words_done:    int = 0
    downloads_done: int = 0

    def set_word(self, word: str) -> None:
        with self.lock:
            self.current_word = word
            self.words_done += 1

    def file_done(self, filename: str) -> None:
        with self.lock:
            self.current_file = filename
            self.downloads_done += 1

    def render(self) -> Table:
        table = Table.grid(padding=(0, 2))
        rows = [
            ("Mot courant",   self.current_word),
            ("Mots traités",  str(self.words_done)),
            ("Téléchargés",   str(self.downloads_done)),
            ("Dernier fichier", self.current_file),
        ]
        for label, value in rows:
            table.add_row(f"[bold]{label}[/bold]", value)
        return table


# ── Téléchargement ────────────────────────────────────────────────────────────

def download_video(
    url: str,
    dest: Path,
    dashboard: Dashboard,
    download_progress: Progress,
    download_task,
    failed: list[str],
) -> None:
    """Télécharge une vidéo vers `dest`. Thread-safe."""
    try:
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)

        with get_session().get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(65_536):
                    if chunk:
                        f.write(chunk)

        dashboard.file_done(dest.name)

    except Exception as exc:
        failed.append(f"{url} -> {dest}  ({exc})")

    finally:
        download_progress.update(download_task, advance=1)


# ── Interface Rich ────────────────────────────────────────────────────────────

def build_layout(
    words_progress: Progress,
    dashboard: Dashboard,
    download_progress: Progress,
) -> Group:
    return Group(
        Panel(words_progress,        title="[bold cyan]Analyse[/bold cyan]"),
        Panel(dashboard.render(),    title="[bold yellow]État[/bold yellow]"),
        Panel(download_progress,     title="[bold green]Téléchargements[/bold green]"),
    )


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    data: dict[str, list[str]] = json.loads(
        Path(DICT_FILE).read_text(encoding="utf-8")
    )
    total_words = sum(len(words) for words in data.values())

    dashboard  = Dashboard()
    failed: list[str] = []
    seen:   set[str]  = set()
    seen_lock = threading.Lock()

    words_progress = Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    download_progress = Progress(
        TextColumn("[green]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TransferSpeedColumn(),
    )

    words_task    = words_progress.add_task("Mots",             total=total_words)
    download_task = download_progress.add_task("Téléchargements", total=0)

    with Live(refresh_per_second=5) as live:
        live.update(build_layout(words_progress, dashboard, download_progress))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []

            for _, words in data.items():
                for word in words:
                    try:
                        ids = get_word_ids(word)
                    except Exception as exc:
                        failed.append(f"[get_ids] {word!r}: {exc}")
                        words_progress.update(words_task, advance=1)
                        continue

                    dest = OUTPUT_DIR / sanitize(word)

                    for word_id in ids:
                        try:
                            video_urls = get_video_urls(word, word_id)
                        except Exception as exc:
                            failed.append(f"[get_urls] {word!r} id={word_id}: {exc}")
                            continue
                        for url in video_urls:
                            with seen_lock:
                                if url in seen:
                                    continue
                                seen.add(url)
                                download_progress.update(download_task, total=len(seen))

                            future = executor.submit(
                                download_video,
                                url,
                                dest / url.split("/")[-1],
                                dashboard,
                                download_progress,
                                download_task,
                                failed,
                            )
                            futures.append(future)

                    dashboard.set_word(word)
                    words_progress.update(words_task, advance=1)
                    live.update(
                        build_layout(words_progress, dashboard, download_progress),
                        refresh=True,
                    )

            # Attendre la fin de tous les téléchargements
            for future in as_completed(futures):
                future.result()

    # ── Rapport final ──────────────────────────────────────────────────────
    if failed:
        print(f"\n[!] {len(failed)} échec(s) :")
        for entry in failed:
            print(f"    {entry}")
    else:
        print("\n✓ Tous les téléchargements ont réussi.")


if __name__ == "__main__":
    main()