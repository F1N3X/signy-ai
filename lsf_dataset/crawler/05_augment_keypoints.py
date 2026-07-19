"""
Augmentation de données avec oversampling ciblé par classe.

Objectif : amener TOUTES les classes à TARGET_PER_CLASS clips exactement,
en générant autant de variantes augmentées que nécessaire par clip source.

Les clips originaux sont toujours inclus.
Les variantes sont des combinaisons aléatoires de 7 transformations.

Augmentations appliquées (toutes sur les coordonnées 3D directement) :
  - Bruit gaussien       : simule imprécision MediaPipe / tremblement
  - Scaling spatial      : personne plus proche ou plus loin
  - Rotation 3D          : inclinaison/rotation du buste et de la tête
  - Translation          : décalage dans le cadre
  - Time warping         : signe exécuté plus vite/lentement (non-uniforme)
  - Frame dropout        : frames aléatoirement zeroed puis interpolées
  - Mirroring horizontal : symétrie G/D (inverse les mains correctement)
  
Structure de sortie :
  lsf_dataset/keypoints_augmented/
    bonjour/
      bonjour_0000_orig.npy
      bonjour_0000_aug000.npy
      ...
    ...
  lsf_dataset/keypoints_augmented/labels.json
"""

import json
import sys
import time
import shutil
from pathlib import Path

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_DIR        = Path("lsf_dataset/keypoints")
OUTPUT_DIR       = Path("lsf_dataset/keypoints_augmented")
TARGET_PER_CLASS = 50
SEED             = 42

# Dimensions (doivent correspondre à 04_extract_keypoints.py)
TARGET_FRAMES = 30
N_FACE        = 73
N_LANDMARKS   = 21 + 21 + 33 + N_FACE   # 148
FEATURE_DIM   = N_LANDMARKS * 3          # 444

# ── Indices des groupes dans le vecteur global ─────────────────────────────────
# [main_G (0:63), main_D (63:126), pose (126:225), visage (225:444)]
POSE_OFFSET = 42   # 21 main_G + 21 main_D

# Paires symétriques dans la pose MediaPipe (index locaux dans le bloc pose)
POSE_MIRROR_PAIRS = [
    (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28),
    (29, 30), (31, 32),
]

# ── Intensités agressives ──────────────────────────────────────────────────────
AUG_CFG = dict(
    noise_std       = 0.035,
    scale_range     = (0.75, 1.30),
    rotation_deg    = 20.0,
    translation_std = 0.15,
    time_warp_std   = 0.35,
    dropout_prob    = 0.15,
    mirror_prob     = 0.5,
)


# ── Transformations ────────────────────────────────────────────────────────────

def aug_noise(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    std = rng.uniform(0, AUG_CFG["noise_std"])
    return (seq + rng.normal(0, std, seq.shape)).astype(np.float32)


def aug_scale(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lo, hi = AUG_CFG["scale_range"]
    return (seq * rng.uniform(lo, hi)).astype(np.float32)


def aug_rotate(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    max_rad = np.deg2rad(AUG_CFG["rotation_deg"])
    ax, ay, az = rng.uniform(-max_rad, max_rad, 3)
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(ax), -np.sin(ax)],
                   [0,  np.sin(ax),  np.cos(ax)]], dtype=np.float32)
    Ry = np.array([[ np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]], dtype=np.float32)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    s = seq.reshape(TARGET_FRAMES, N_LANDMARKS, 3)
    return (s @ R.T).reshape(TARGET_FRAMES, FEATURE_DIM).astype(np.float32)


def aug_translate(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    delta = rng.normal(0, AUG_CFG["translation_std"], (1, 3)).astype(np.float32)
    s = seq.reshape(TARGET_FRAMES, N_LANDMARKS, 3) + delta
    return s.reshape(TARGET_FRAMES, FEATURE_DIM).astype(np.float32)


def aug_time_warp(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    T       = TARGET_FRAMES
    std     = AUG_CFG["time_warp_std"]
    knot_x  = np.linspace(0, T - 1, 5)
    knot_y  = knot_x + rng.normal(0, std * T, 5)
    knot_y[0], knot_y[-1] = 0, T - 1
    knot_y  = np.clip(knot_y, 0, T - 1)
    idx     = np.interp(np.arange(T), knot_x, knot_y)
    left    = np.floor(idx).astype(int)
    right   = np.clip(left + 1, 0, T - 1)
    alpha   = (idx - left)[:, None]
    return ((1 - alpha) * seq[left] + alpha * seq[right]).astype(np.float32)


def aug_dropout(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    T      = TARGET_FRAMES
    mask   = rng.random(T) > AUG_CFG["dropout_prob"]
    if mask.all():
        return seq.copy()
    valid  = np.where(mask)[0]
    if len(valid) == 0:
        return seq.copy()
    result = seq.copy()
    for i in range(T):
        if not mask[i]:
            before = valid[valid < i]
            after  = valid[valid > i]
            if len(before) == 0:
                result[i] = seq[after[0]]
            elif len(after) == 0:
                result[i] = seq[before[-1]]
            else:
                b, a  = before[-1], after[0]
                alpha = (i - b) / (a - b)
                result[i] = (1 - alpha) * seq[b] + alpha * seq[a]
    return result.astype(np.float32)


def aug_mirror(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() > AUG_CFG["mirror_prob"]:
        return seq.copy()
    s = seq.reshape(TARGET_FRAMES, N_LANDMARKS, 3).copy()
    s[:, :, 0] = -s[:, :, 0]                       # inverser axe X
    s[:, :21, :], s[:, 21:42, :] = (                # échanger mains
        s[:, 21:42, :].copy(), s[:, :21, :].copy()
    )
    for l, r in POSE_MIRROR_PAIRS:                  # paires symétriques pose
        tmp = s[:, POSE_OFFSET + l, :].copy()
        s[:, POSE_OFFSET + l, :] = s[:, POSE_OFFSET + r, :]
        s[:, POSE_OFFSET + r, :] = tmp
    return s.reshape(TARGET_FRAMES, FEATURE_DIM).astype(np.float32)


ALL_TRANSFORMS = [
    aug_noise, aug_scale, aug_rotate,
    aug_translate, aug_time_warp, aug_dropout, aug_mirror,
]


def augment_one(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applique une combinaison aléatoire de transformations (proba 0.7 chacune)."""
    result = seq.copy()
    for fn in ALL_TRANSFORMS:
        if fn is aug_mirror or rng.random() < 0.7:
            result = fn(result, rng)
    return result


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copier labels.json
    src_labels = INPUT_DIR / "labels.json"
    if src_labels.exists():
        shutil.copy(src_labels, OUTPUT_DIR / "labels.json")
        labels = json.loads(src_labels.read_text())
    else:
        print("labels.json introuvable dans", INPUT_DIR)
        return

    # Inventaire des clips par classe
    clips_by_class: dict[str, list[Path]] = {}
    for word in sorted(labels):
        word_dir = INPUT_DIR / word
        if not word_dir.exists():
            clips_by_class[word] = []
            continue
        clips_by_class[word] = sorted(word_dir.glob("*.npy"))

    # Afficher le plan d'augmentation
    print(f"{'Classe':<25} {'Originaux':>9} {'→ Générés':>9} {'Ratio':>7}")
    print("─" * 54)
    total_out = 0
    for word, clips in clips_by_class.items():
        n      = len(clips)
        needed = max(TARGET_PER_CLASS - n, 0)
        ratio  = TARGET_PER_CLASS / n if n > 0 else float("inf")
        print(f"{word:<25} {n:>9} {TARGET_PER_CLASS:>9} {ratio:>6.1f}×")
        total_out += TARGET_PER_CLASS
    print("─" * 54)
    print(f"{'TOTAL':<25} {sum(len(v) for v in clips_by_class.values()):>9} {total_out:>9}\n")

    # Génération
    done  = 0
    start = time.time()
    bar_w = 28

    for word, src_clips in clips_by_class.items():
        out_dir = OUTPUT_DIR / word
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src_clips:
            print(f"\n[!] {word} : aucun clip source, classe ignorée.")
            continue

        # 1. Copier tous les originaux
        for src in src_clips:
            shutil.copy(src, out_dir / f"{src.stem}_orig.npy")
            done += 1
            _bar(done, total_out, src.stem, start, bar_w)

        # 2. Générer des variantes jusqu'à TARGET_PER_CLASS
        needed    = TARGET_PER_CLASS - len(src_clips)
        aug_count = 0
        # Cycler sur les clips sources pour distribuer équitablement
        src_cycle = [np.load(p) for p in src_clips]
        src_idx   = 0

        while aug_count < needed:
            seq  = src_cycle[src_idx % len(src_cycle)]
            stem = src_clips[src_idx % len(src_clips)].stem
            aug  = augment_one(seq, rng)
            name = f"{stem}_aug{aug_count:03d}.npy"
            np.save(out_dir / name, aug)
            aug_count += 1
            src_idx   += 1
            done      += 1
            _bar(done, total_out, name, start, bar_w)

    sys.stdout.write("\n")
    elapsed = time.time() - start
    print(f"\n✓ {total_out} clips dans {OUTPUT_DIR}  ({elapsed:.1f}s)")
    print(f"  Toutes les classes : {TARGET_PER_CLASS} clips")
    print(f"  Dimension conservée : ({TARGET_FRAMES}, {FEATURE_DIM})")


def _bar(done: int, total: int, label: str, start: float, w: int) -> None:
    pct     = done / total
    filled  = int(w * pct)
    bar     = "█" * filled + "░" * (w - filled)
    elapsed = time.time() - start
    eta     = (elapsed / pct - elapsed) if pct > 0 else 0
    name    = label[:30].ljust(30)
    sys.stdout.write(
        f"\r[{bar}] {done:>{len(str(total))}}/{total}"
        f"  {pct*100:5.1f}%  {name}"
        f"  ETA {int(eta//60):02d}:{int(eta%60):02d}"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()