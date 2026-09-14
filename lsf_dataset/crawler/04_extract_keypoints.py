"""
Extraction et normalisation des keypoints MediaPipe depuis les vidéos LSF.
API MediaPipe >= 0.10 (PoseLandmarker + HandLandmarker + FaceLandmarker)

Pour chaque vidéo :
  1. Les trois landmarkers extraient pose, mains et visage frame par frame
  2. Les coordonnées sont normalisées (origine = milieu épaules, scale = largeur épaules)
  3. Le clip est rééchantillonné à TARGET_FRAMES frames
  4. Le résultat est sauvegardé en .npy dans OUTPUT_DIR

Structure de sortie :
  lsf_dataset/keypoints/
    bonjour/
      bonjour_0000.npy   # shape (TARGET_FRAMES, FEATURE_DIM)
    merci/
      ...
  lsf_dataset/keypoints/labels.json

Modèles requis (à télécharger une fois) :
  mkdir -p lsf_dataset/models
  wget -O lsf_dataset/models/pose.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
  wget -O lsf_dataset/models/hand.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  wget -O lsf_dataset/models/face.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
"""

import os

# Silencer les logs C++ de TensorFlow/MediaPipe AVANT tout import
# 0=tous, 1=info, 2=warning, 3=erreur seulement
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["GLOG_minloglevel"]       = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"   # evite les logs GL/EGL
os.environ["GRPC_VERBOSITY"]         = "ERROR"

import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

VIDEO_DIR     = Path("lsf_dataset/to_train_videos")
OUTPUT_DIR    = Path("lsf_dataset/keypoints")
MODELS_DIR    = Path("lsf_dataset/models")

TARGET_FRAMES = 90    # frames par clip après rééchantillonnage
MAX_WORKERS   = 1     # MediaPipe tasks ne sont pas fork-safe → laisser à 1
MIN_FRAMES    = 5     # clips trop courts → ignorés

# ── Sélection des landmarks du visage (sur 478 dans FaceLandmarker) ────────────
#
# On garde ~70 points clés : lèvres, sourcils, yeux, nez, contour
#
FACE_LANDMARKS = [
    # Lèvres extérieures
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    # Lèvres intérieures
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    # Sourcil gauche
    70, 63, 105, 66, 107,
    # Sourcil droit
    336, 296, 334, 293, 300,
    # Œil gauche
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    # Œil droit
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    # Nez
    1, 2, 5, 4, 6, 19, 94,
    # Contour du visage
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    127, 162, 21, 54, 103, 67, 109,
]
N_FACE = len(FACE_LANDMARKS)

# Dimensions du vecteur final par frame :
# 21 (main G) + 21 (main D) + 33 (pose) + N_FACE (visage), chacun × 3 coords
N_LANDMARKS = 21 + 21 + 33 + N_FACE
FEATURE_DIM = N_LANDMARKS * 3

# ── Indices pour la normalisation ─────────────────────────────────────────────
# Pose : épaule gauche = index 11, épaule droite = index 12
# Dans le vecteur global [main_G, main_D, pose, visage] :
POSE_OFFSET        = 21 + 21
SHOULDER_LEFT_IDX  = POSE_OFFSET + 11
SHOULDER_RIGHT_IDX = POSE_OFFSET + 12


# ── Construction des landmarkers ──────────────────────────────────────────────

def _make_landmarkers():
    """Instancie les trois landmarkers MediaPipe 0.10+."""
    BaseOptions = mp_python.BaseOptions
    RunningMode = mp_vision.RunningMode

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "pose.task")),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "hand.task")),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "face.task")),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return (
        mp_vision.PoseLandmarker.create_from_options(pose_opts),
        mp_vision.HandLandmarker.create_from_options(hand_opts),
        mp_vision.FaceLandmarker.create_from_options(face_opts),
    )


# ── Extraction d'un clip ───────────────────────────────────────────────────────

def extract_clip(video_path: Path) -> np.ndarray | None:
    """
    Extrait les keypoints d'une vidéo.
    Retourne un array (TARGET_FRAMES, FEATURE_DIM) ou None si échec.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames_landmarks = []

    pose_lm, hand_lm, face_lm = _make_landmarkers()

    try:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # MediaPipe 0.10 attend un objet mp.Image avec timestamp en ms
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)

            pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_lm.detect_for_video(mp_image, timestamp_ms)
            face_result = face_lm.detect_for_video(mp_image, timestamp_ms)

            row = _landmarks_to_row(pose_result, hand_result, face_result)
            frames_landmarks.append(row)
            frame_idx += 1

    finally:
        cap.release()
        pose_lm.close()
        hand_lm.close()
        face_lm.close()

    if len(frames_landmarks) < MIN_FRAMES:
        return None

    sequence = np.array(frames_landmarks, dtype=np.float32)
    sequence = _normalize(sequence)
    sequence = _resample(sequence, TARGET_FRAMES)
    return sequence


def _landmarks_to_row(pose_result, hand_result, face_result) -> np.ndarray:
    """
    Convertit les résultats d'une frame en vecteur plat.
    Ordre : [main_G (21×3), main_D (21×3), pose (33×3), visage (N_FACE×3)]
    Points non détectés → 0.0
    """
    # ── Pose (33 points) ──────────────────────────────────────────────────────
    if pose_result.pose_landmarks:
        lms  = pose_result.pose_landmarks[0]
        pose = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                        dtype=np.float32).flatten()
    else:
        pose = np.zeros(33 * 3, dtype=np.float32)

    # ── Mains (21 points chacune) ─────────────────────────────────────────────
    # HandLandmarker retourne jusqu'à 2 mains avec leur handedness
    left_hand  = np.zeros(21 * 3, dtype=np.float32)
    right_hand = np.zeros(21 * 3, dtype=np.float32)

    if hand_result.hand_landmarks:
        for i, hand_lms in enumerate(hand_result.hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms],
                           dtype=np.float32).flatten()
            # handedness : "Left" ou "Right" (depuis le point de vue de la caméra)
            label = hand_result.handedness[i][0].category_name
            if label == "Left":
                left_hand = pts
            else:
                right_hand = pts

    # ── Visage (N_FACE points sélectionnés) ──────────────────────────────────
    if face_result.face_landmarks:
        all_face = face_result.face_landmarks[0]
        pts = [[all_face[i].x, all_face[i].y, all_face[i].z]
               for i in FACE_LANDMARKS]
        face = np.array(pts, dtype=np.float32).flatten()
    else:
        face = np.zeros(N_FACE * 3, dtype=np.float32)

    return np.concatenate([left_hand, right_hand, pose, face])


def _normalize(sequence: np.ndarray) -> np.ndarray:
    """
    Normalise par rapport aux épaules.
    Origine = milieu épaules | Échelle = distance inter-épaules.
    Forward-fill si les épaules sont absentes sur une frame.
    """
    seq = sequence.reshape(len(sequence), N_LANDMARKS, 3).copy()

    last_origin = None
    last_scale  = None

    for i, frame in enumerate(seq):
        ls = frame[SHOULDER_LEFT_IDX]
        rs = frame[SHOULDER_RIGHT_IDX]
        detected = not (np.allclose(ls, 0) and np.allclose(rs, 0))

        if detected:
            origin = (ls + rs) / 2.0
            scale  = np.linalg.norm(ls - rs)
            scale  = scale if scale > 1e-6 else 1.0
            last_origin, last_scale = origin, scale
        elif last_origin is not None:
            origin, scale = last_origin, last_scale
        else:
            continue

        seq[i] = (frame - origin) / scale

    return seq.reshape(len(sequence), FEATURE_DIM)


def _resample(sequence: np.ndarray, target: int) -> np.ndarray:
    """Interpolation linéaire pour ramener le clip à `target` frames."""
    T = len(sequence)
    if T == target:
        return sequence
    idx   = np.linspace(0, T - 1, target)
    left  = np.floor(idx).astype(int)
    right = np.clip(left + 1, 0, T - 1)
    alpha = (idx - left)[:, None]
    return (1 - alpha) * sequence[left] + alpha * sequence[right]


# ── Worker (gardé séparé pour faciliter le passage futur en multiprocess) ─────

def _worker(args: tuple[Path, Path]) -> tuple[str, bool, str]:
    video_path, out_path = args
    try:
        kp = extract_clip(video_path)
        if kp is None:
            return str(video_path), False, "clip trop court ou illisible"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, kp)
        return str(video_path), True, ""
    except Exception as e:
        return str(video_path), False, str(e)


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    # Vérifier que les modèles sont présents
    for name in ("pose.task", "hand.task", "face.task"):
        p = MODELS_DIR / name
        if not p.exists():
            print(f"[!] Modèle manquant : {p}")
            print("    Voir les commandes wget en haut de ce fichier.")
            return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks:     list[tuple[Path, Path]] = []
    label_set: set[str] = set()

    for word_dir in sorted(VIDEO_DIR.iterdir()):
        if not word_dir.is_dir():
            continue
        word = word_dir.name
        label_set.add(word)
        video_files = sorted(
            video_file
            for video_file in word_dir.iterdir()
            if video_file.is_file() and video_file.suffix.lower() in {".mp4", ".webm"}
        )
        for i, video_file in enumerate(video_files):
            out_path = OUTPUT_DIR / word / f"{word}_{i:04d}.npy"
            if out_path.exists():
                continue
            tasks.append((video_file, out_path))

    if not tasks:
        print("Rien à extraire (tout est déjà fait ou dossier vide).")
        return

    labels = {word: idx for idx, word in enumerate(sorted(label_set))}
    labels_path = OUTPUT_DIR / "labels.json"
    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    print(f"{len(labels)} classes, {len(tasks)} vidéos à traiter.\n")

    failed: list[tuple[str, str]] = []
    start = time.time()

    for i, t in enumerate(tasks, 1):
        path, ok, reason = _worker(t)
        if not ok:
            failed.append((path, reason))

        # Barre de progression inline (écrase la ligne précédente)
        done_n  = i
        total_n = len(tasks)
        pct     = done_n / total_n
        bar_w   = 30
        filled  = int(bar_w * pct)
        bar     = "█" * filled + "░" * (bar_w - filled)
        elapsed = time.time() - start
        eta     = (elapsed / pct - elapsed) if pct > 0 else 0
        label   = Path(path).name[:35].ljust(35)

        sys.stdout.write(
            f"\r[{bar}] {done_n:>{len(str(total_n))}}/{total_n}"
            f"  {pct*100:5.1f}%"
            f"  {label}"
            f"  ETA {int(eta//60):02d}:{int(eta%60):02d}"
            + ("  ✗" if not ok else "   ")
        )
        sys.stdout.flush()

    sys.stdout.write("\n")  # sortir de la ligne de progression
    done = len(tasks) - len(failed)
    print(f"✓ {done}/{len(tasks)} clips extraits.")

    if failed:
        print(f"\n[!] {len(failed)} échec(s) :")
        for path, reason in failed[:20]:
            print(f"    {path}  →  {reason}")
        if len(failed) > 20:
            print(f"    ... et {len(failed) - 20} autres.")

    print(f"\nDimension d'un clip : ({TARGET_FRAMES}, {FEATURE_DIM})")
    print(f"  = {TARGET_FRAMES} frames × {N_LANDMARKS} landmarks × 3 coords")
    print(f"  dont {N_FACE} landmarks de visage sélectionnés")
    print(f"\nLabels : {labels_path}")


if __name__ == "__main__":
    main()