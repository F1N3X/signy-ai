"""
Entraînement du modèle de reconnaissance de signes LSF.

Architecture : Transformer Encoder + classification head
  - Input  : séquences de keypoints (T frames × FEATURE_DIM coords)
  - Output : classe du signe (N_CLASSES)

Reprend les conventions du notebook CNN précédent :
  - TensorBoard, EarlyStopping, torchmetrics
  - Export ONNX en fin d'entraînement
  - Matrice de confusion + métriques par classe

Usage :
  pip install torch torchmetrics tqdm --break-system-packages
  python lsf_dataset/crawler/06_train.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")   # pas de GUI sur le serveur
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR      = Path("lsf_dataset/keypoints_augmented")
OUTPUT_DIR    = Path("lsf_dataset/model")

TARGET_FRAMES = 90
FEATURE_DIM   = 444     # 148 landmarks × 3 coords

# Hyperparamètres
BATCH_SIZE    = 32
EPOCHS        = 100
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.2
PATIENCE      = 15      # early stopping

# Architecture Transformer
D_MODEL       = 64     # dimension d'embedding
N_HEADS       = 4       # têtes d'attention (D_MODEL doit être divisible)
N_LAYERS      = 3       # couches encoder
DIM_FF        = 256     # dimension feed-forward interne

# Split
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST_RATIO  = 0.15 (le reste)

SEED          = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ── Dataset ────────────────────────────────────────────────────────────────────

class KeypointDataset(Dataset):
    """
    Charge les fichiers .npy depuis DATA_DIR/<classe>/<clip>.npy
    Shape de chaque sample : (TARGET_FRAMES, FEATURE_DIM)
    """
    def __init__(self, data_dir: Path, labels: dict[str, int]):
        self.samples: list[tuple[Path, int]] = []
        for word, idx in labels.items():
            word_dir = data_dir / word
            if not word_dir.exists():
                continue
            for npy in sorted(word_dir.glob("*.npy")):
                self.samples.append((npy, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[i]
        seq = np.load(path).astype(np.float32)   # (T, FEATURE_DIM)
        return torch.from_numpy(seq), label

    def class_weights(self, n_classes: int) -> torch.Tensor:
        """Poids inversement proportionnels à la fréquence de chaque classe."""
        counts = torch.zeros(n_classes)
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.sum()


# ── Modèle ─────────────────────────────────────────────────────────────────────

class LSFTransformer(nn.Module):
    """
    Transformer Encoder pour la reconnaissance de signes LSF.

    Pipeline :
      1. Projection linéaire FEATURE_DIM → D_MODEL
      2. Positional encoding sinusoïdal (injection de l'ordre temporel)
      3. N couches Transformer Encoder (attention multi-têtes + FFN)
      4. Agrégation temporelle (mean pooling sur les T frames)
      5. Classification head → N_CLASSES logits
    """
    def __init__(self, n_classes: int):
        super().__init__()

        # 1. Projection d'entrée
        self.input_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, D_MODEL),
            nn.LayerNorm(D_MODEL),
        )

        # 2. Positional encoding (fixe, sinusoïdal)
        self.register_buffer("pos_enc", self._make_pos_enc(TARGET_FRAMES, D_MODEL))

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=True,   # (batch, seq, features)
            norm_first=True,    # Pre-LN : plus stable à l'entraînement
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # 4+5. Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT / 2),
            nn.Linear(D_MODEL // 2, n_classes),
        )

    @staticmethod
    def _make_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        """Positional encoding sinusoïdal standard (Vaswani et al. 2017)."""
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, T, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, T, FEATURE_DIM)
        x = self.input_proj(x)          # → (batch, T, D_MODEL)
        x = x + self.pos_enc            # + positional encoding
        x = self.encoder(x)             # → (batch, T, D_MODEL)
        x = x.mean(dim=1)              # mean pooling → (batch, D_MODEL)
        return self.classifier(x)       # → (batch, N_CLASSES)


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4, path: Path = None):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = path
        self.counter   = 0
        self.best_loss = None
        self.stop      = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            if self.path:
                torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ── Boucles d'entraînement / évaluation ───────────────────────────────────────

def train_epoch(loader, model, criterion, optimizer, writer, global_step):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    loop = tqdm(loader, desc="Train", leave=False)
    for X, y in loop:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs      = X.size(0)
        correct = (logits.argmax(1) == y).sum().item()
        total_loss    += loss.item() * bs
        total_correct += correct
        total_samples += bs

        writer.add_scalar("Batch/Loss",     loss.item(),      global_step)
        writer.add_scalar("Batch/Accuracy", correct / bs,     global_step)
        global_step += 1

        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/bs:.1f}%")

    return total_loss / total_samples, total_correct / total_samples, global_step


def eval_epoch(loader, model, criterion, n_classes):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    acc_m  = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(device)
    prec_m = torchmetrics.Precision(task="multiclass", num_classes=n_classes, average="macro").to(device)
    rec_m  = torchmetrics.Recall(task="multiclass", num_classes=n_classes, average="macro").to(device)
    f1_m   = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average="macro").to(device)
    cm_m   = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=n_classes).to(device)

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Eval", leave=False):
            X, y   = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)
            preds  = logits.argmax(1)

            bs = X.size(0)
            total_loss    += loss.item() * bs
            total_correct += (preds == y).sum().item()
            total_samples += bs

            acc_m.update(preds, y);  prec_m.update(preds, y)
            rec_m.update(preds, y);  f1_m.update(preds, y)
            cm_m.update(preds, y)

    metrics = {
        "loss":      total_loss / total_samples,
        "accuracy":  acc_m.compute().item(),
        "precision": prec_m.compute().item(),
        "recall":    rec_m.compute().item(),
        "f1":        f1_m.compute().item(),
        "confmat":   cm_m.compute().cpu(),
    }
    for m in [acc_m, prec_m, rec_m, f1_m, cm_m]:
        m.reset()
    return metrics


# ── Visualisations ─────────────────────────────────────────────────────────────

def plot_history(history: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"],   marker="o", label="val")
    axes[0].set_title("Loss par epoch"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"],   marker="o", label="val")
    axes[1].set_title("Accuracy par epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)


def plot_confusion_matrix(confmat: torch.Tensor, class_names: list[str], out_dir: Path) -> None:
    cm  = confmat.numpy()
    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n)))
    im  = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matrice de confusion — Test")
    ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe vraie")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    thresh = cm.max() * 0.6
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def plot_per_class_metrics(confmat: torch.Tensor, class_names: list[str], out_dir: Path) -> None:
    cm   = confmat.numpy().astype(np.float64)
    tp   = np.diag(cm)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec  = tp / np.maximum(cm.sum(axis=1), 1)
    f1   = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    n    = len(class_names)
    x    = np.arange(n); w = 0.25

    fig, ax = plt.subplots(figsize=(max(12, n * 1.2), 5))
    ax.bar(x - w, prec, w, label="Precision")
    ax.bar(x,     rec,  w, label="Recall")
    ax.bar(x + w, f1,   w, label="F1-score")
    ax.set_title("Métriques par classe — Test")
    ax.set_ylim(0, 1.05); ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_metrics.png", dpi=120)
    plt.close(fig)


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Labels
    labels_path = DATA_DIR / "labels.json"
    if not labels_path.exists():
        print(f"[!] labels.json introuvable dans {DATA_DIR}"); return
    labels      = json.loads(labels_path.read_text(encoding="utf-8"))
    n_classes   = len(labels)
    class_names = [k for k, _ in sorted(labels.items(), key=lambda x: x[1])]
    print(f"{n_classes} classes : {class_names}")

    # Dataset + split 70/15/15
    full_ds = KeypointDataset(DATA_DIR, labels)
    print(f"Total clips : {len(full_ds)}")

    total      = len(full_ds)
    train_size = int(TRAIN_RATIO * total)
    val_size   = int(VAL_RATIO   * total)
    test_size  = total - train_size - val_size

    ds_train, ds_val, ds_test = random_split(
        full_ds, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Train : {len(ds_train)} | Val : {len(ds_val)} | Test : {len(ds_test)}")

    # WeightedRandomSampler pour compenser les éventuels déséquilibres résiduels
    sample_weights = [full_ds.class_weights(n_classes)[label].item()
                      for _, label in [full_ds.samples[i] for i in ds_train.indices]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(ds_train), replacement=True)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=2, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=2, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False,    num_workers=2, pin_memory=True)

    # Modèle
    model = LSFTransformer(n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {n_params:,}")
    print(model)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 20)

    # TensorBoard + EarlyStopping
    writer  = SummaryWriter(log_dir=str(OUTPUT_DIR / "runs"))
    stopper = EarlyStopping(patience=PATIENCE, path=OUTPUT_DIR / "best_model.pt")

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    global_step = 0

    print(f"\nEntraînement sur {EPOCHS} epochs max (early stopping patience={PATIENCE})\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc, global_step = train_epoch(
            dl_train, model, criterion, optimizer, writer, global_step
        )
        val_metrics = eval_epoch(dl_val, model, criterion, n_classes)
        val_loss    = val_metrics["loss"]
        val_acc     = val_metrics["accuracy"]

        scheduler.step()
        stopper(val_loss, model)

        # Logs
        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss);     history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        flag    = " ✓ best" if stopper.counter == 0 else f" ({stopper.counter}/{PATIENCE})"
        print(
            f"Epoch {epoch:3d}/{EPOCHS}"
            f"  train_loss={train_loss:.4f}  train_acc={100*train_acc:.1f}%"
            f"  val_loss={val_loss:.4f}  val_acc={100*val_acc:.1f}%"
            f"  {elapsed:.1f}s{flag}"
        )

        if stopper.stop:
            print(f"\nEarly stopping à l'epoch {epoch}.")
            break

    writer.close()

    # ── Évaluation finale sur le test set ─────────────────────────────────────
    print("\nChargement du meilleur modèle pour évaluation finale...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", map_location=device))
    test_metrics = eval_epoch(dl_test, model, criterion, n_classes)

    print(f"\n── Résultats Test ──────────────────────────────")
    print(f"  Loss      : {test_metrics['loss']:.4f}")
    print(f"  Accuracy  : {100*test_metrics['accuracy']:.2f}%")
    print(f"  Precision : {100*test_metrics['precision']:.2f}%")
    print(f"  Recall    : {100*test_metrics['recall']:.2f}%")
    print(f"  F1 macro  : {100*test_metrics['f1']:.2f}%")
    print(f"────────────────────────────────────────────────")

    # ── Graphiques ─────────────────────────────────────────────────────────────
    plot_history(history, OUTPUT_DIR)
    plot_confusion_matrix(test_metrics["confmat"], class_names, OUTPUT_DIR)
    plot_per_class_metrics(test_metrics["confmat"], class_names, OUTPUT_DIR)
    print(f"\nGraphiques sauvegardés dans {OUTPUT_DIR}/")

    # ── Export ONNX ────────────────────────────────────────────────────────────
    model.eval()
    dummy = torch.zeros(1, TARGET_FRAMES, FEATURE_DIM, device=device)
    onnx_path = OUTPUT_DIR / "lsf_model.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["keypoints"],
        output_names=["logits"],
        dynamic_axes={"keypoints": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"✓ Modèle exporté : {onnx_path}")

    # ── Sauvegarde des métadonnées ─────────────────────────────────────────────
    meta = {
        "classes":      class_names,
        "n_classes":    n_classes,
        "feature_dim":  FEATURE_DIM,
        "target_frames": TARGET_FRAMES,
        "d_model":      D_MODEL,
        "n_heads":      N_HEADS,
        "n_layers":     N_LAYERS,
    }
    (OUTPUT_DIR / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    print(f"✓ Métadonnées sauvegardées : {OUTPUT_DIR / 'model_meta.json'}")


if __name__ == "__main__":
    main()
