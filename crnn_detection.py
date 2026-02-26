"""
European License Plate Recognition System – CRNN with CTC
==========================================================
A CRNN-based end-to-end plate text recognizer that reads entire plate
strips without segmenting individual characters.

Dataset  : https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset
           (identical source as cnn_detection.py)

Pipeline :
    1. Download dataset via kagglehub (same helper as CNN)
    2. Run build_char_dataset() for segmentation stats (shared with CNN)
    3. Collect all valid plate images and labels
    4. Split plates 80 / 20 into train / test
    5. Carve out 15 % of train as validation
    6. Train a CRNN (CNN + BiLSTM + CTC) on full plate strips
    7. Evaluate: character-level and plate-level metrics
    8. Export crnn_metrics.csv (same columns as cnn_metrics.csv)
    9. Plot loss curves → loss_curves_crnn.png
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import logging
import random
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Re-use shared components from cnn_detection (DO NOT rewrite / simplify)
# ---------------------------------------------------------------------------
from cnn_detection import (
    # Classes
    CharacterSegmenter,
    # Functions
    build_char_dataset,
    extract_label_from_filename,
    edit_distance,
    save_metrics_csv,
    download_dataset,
    find_dataset_root,
    get_device,
    plot_loss_curves,
    # Character helpers
    char_to_idx,
    idx_to_char,
    # Constants  – must be identical across CNN / CRNN
    IMG_SIZE,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DROPOUT_RATE,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,          # not used by CTC, kept for documentation
    EARLY_STOP_PATIENCE,
    SEED,
    MIN_PLATE_CHARS,
    MAX_PLATE_CHARS,
    VALID_CHARS,
    CHAR_TO_IDX,
    IDX_TO_CHAR,
    # Segmenter construction constants
    MIN_CHAR_HEIGHT_RATIO,
    MIN_CHAR_AREA,
    MAX_CHAR_WIDTH_RATIO,
    NMS_IOU_THRESHOLD,
    MIN_CHAR_ASPECT,
    MAX_CHAR_ASPECT,
    VERT_CENTER_TOL,
    BORDER_EXCLUSION_RATIO,
    MIN_WIDTH_RATIO_VS_MEDIAN,
    MIN_HEIGHT_RATIO_VS_MEDIAN,
    MAX_HEIGHT_RATIO_VS_MEDIAN,
)

# ---------------------------------------------------------------------------
# CRNN-specific configuration
# ---------------------------------------------------------------------------
PLATE_W: int = 192                   # wider plate strip -> more CTC timesteps (48 vs 32)
BLANK_IDX: int = NUM_CLASSES         # CTC blank token (index 36)
NUM_CTC_CLASSES: int = NUM_CLASSES + 1   # 36 chars + 1 blank = 37
RNN_HIDDEN: int = 256
RNN_LAYERS: int = 2
CRNN_EPOCHS: int = 200               # same as CNN for fair comparison
CRNN_LR: float = 1e-3                # same as CNN (LEARNING_RATE) for fair comparison
BEAM_WIDTH: int = 10                 # beam search decoding width

# Reproducibility (same seed as CNN)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Plate-level dataset builder
# ---------------------------------------------------------------------------

def build_plate_dataset(
    dataset_root: Path,
    subfolders: Tuple[str, ...] = ("train", "val", "test"),
    img_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> Tuple[List[Path], List[str]]:
    """
    Collect all plate image paths and their cleaned text labels.

    Only keeps plates that:
      - have a readable image (cv2.imread succeeds)
      - have a valid alphanumeric label in [MIN_PLATE_CHARS, MAX_PLATE_CHARS]
    """
    paths: List[Path] = []
    labels: List[str] = []

    for sub in subfolders:
        folder = dataset_root / sub
        if not folder.exists():
            log.warning("Subfolder not found, skipping: %s", folder)
            continue

        for p in sorted(folder.iterdir()):          # sorted for reproducibility
            if p.suffix.lower() not in img_extensions:
                continue
            label = extract_label_from_filename(p)
            label_chars = [ch for ch in label if char_to_idx(ch) is not None]
            clean_label = "".join(label_chars)
            if not (MIN_PLATE_CHARS <= len(clean_label) <= MAX_PLATE_CHARS):
                continue
            # quick readability check (small cost for a dataset of ~700 plates)
            if cv2.imread(str(p)) is None:
                continue
            paths.append(p)
            labels.append(clean_label)

    log.info("Plate dataset: %d valid plates collected.", len(paths))
    return paths, labels


# ---------------------------------------------------------------------------
# 2. PyTorch Dataset for CRNN (full plate strips)
# ---------------------------------------------------------------------------

class PlateStripDataset(Dataset):
    """
    Loads full plate images as fixed-size grayscale strips for CRNN.

    Each sample is a tuple  (image_tensor, encoded_label, label_length).
    Augmentation mirrors ``PlateCharDataset`` in cnn_detection.py.
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[str],
        plate_w: int = PLATE_W,
        img_h: int = IMG_SIZE,
        augment: bool = False,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.plate_w = plate_w
        self.img_h = img_h
        self.augment = augment

        # ---------- same augmentation pipeline as PlateCharDataset ----------
        base_tf = [
            transforms.ToPILImage(),
            transforms.ToTensor(),                        # → [0, 1]
            transforms.Normalize((0.5,), (0.5,)),         # → [-1, 1]
        ]
        aug_tf = [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                shear=5,
            ),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(
                p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0,
            ),
        ]
        self.tf_base = transforms.Compose(base_tf)
        self.tf_aug = transforms.Compose(aug_tf)

    # ------------------------------------------------------------------
    def _load_and_resize(self, idx: int) -> np.ndarray:
        """Load a plate image -> grayscale -> CLAHE -> resize to (img_h, plate_w)."""
        bgr = cv2.imread(str(self.image_paths[idx]))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # CLAHE contrast normalization (adaptive to image contrast)
        std = float(np.std(gray))
        clip = 4.0 if std < 30 else (1.0 if std > 80 else 2.0)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        h, w = gray.shape

        # Resize height to img_h, keep aspect ratio for width
        new_w = max(1, int(w * self.img_h / h))
        resized = cv2.resize(gray, (new_w, self.img_h),
                             interpolation=cv2.INTER_AREA)

        if new_w < self.plate_w:
            # Pad right with zeros (black)
            padded = np.zeros((self.img_h, self.plate_w), dtype=np.uint8)
            padded[:, :new_w] = resized
            return padded
        else:
            # Resize to exact plate_w (slight squeeze)
            return cv2.resize(resized, (self.plate_w, self.img_h),
                              interpolation=cv2.INTER_AREA)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self._load_and_resize(idx)                 # (H, W) uint8
        tf = self.tf_aug if self.augment else self.tf_base
        tensor = tf(img)                                 # (1, H, W) float32

        label_encoded = [CHAR_TO_IDX[ch] for ch in self.labels[idx]]
        return tensor, label_encoded, len(label_encoded)


def ctc_collate_fn(batch):
    """Custom collate that flattens variable-length labels for CTCLoss."""
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, dim=0)                  # (N, 1, H, W)
    targets = torch.cat(
        [torch.tensor(l, dtype=torch.long) for l in labels]
    )
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, targets, label_lengths


def make_crnn_dataloaders(
    paths_train: List[Path],
    labels_train: List[str],
    paths_test: List[Path],
    labels_test: List[str],
    batch_size: int = BATCH_SIZE,
    val_fraction: float = 0.15,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / validation / test DataLoaders for CRNN.

    The training portion is further split with ``val_fraction`` carved out
    for online validation (identical to ``make_dataloaders`` in cnn_detection).
    """
    indices = list(range(len(paths_train)))

    # Stratify by label length as the plate-level analogue of char-class
    lengths = [len(l) for l in labels_train]
    try:
        idx_tr, idx_val = train_test_split(
            indices, test_size=val_fraction,
            random_state=seed, stratify=lengths,
        )
    except ValueError:
        idx_tr, idx_val = train_test_split(
            indices, test_size=val_fraction, random_state=seed,
        )

    paths_tr   = [paths_train[i] for i in idx_tr]
    labels_tr  = [labels_train[i] for i in idx_tr]
    paths_val  = [paths_train[i] for i in idx_val]
    labels_val = [labels_train[i] for i in idx_val]

    train_ds = PlateStripDataset(paths_tr,   labels_tr,   augment=True)
    val_ds   = PlateStripDataset(paths_val,  labels_val,  augment=False)
    test_ds  = PlateStripDataset(paths_test, labels_test, augment=False)

    kw = dict(num_workers=0, pin_memory=False, collate_fn=ctc_collate_fn)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **kw)

    log.info(
        "CRNN DataLoaders | train=%d  val=%d  test=%d  (batch=%d)",
        len(train_ds), len(val_ds), len(test_ds), batch_size,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 3. CRNN Model
# ---------------------------------------------------------------------------

class CRNNBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU with configurable pooling kernel."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        pool_kernel: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool_kernel is not None:
            layers.append(nn.MaxPool2d(kernel_size=pool_kernel,
                                       stride=pool_kernel))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CharCRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) for plate text recognition.

    Architecture (input: 1 x 28 x 192)
    ------------------------------------
    CNN Encoder (wider channels for better feature extraction):
        CRNNBlock 1 : 1   -> 64   + MaxPool(2,2)  ->  (64, 14, 96)
        CRNNBlock 2 : 64  -> 128  + MaxPool(2,2)  -> (128,  7, 48)
        CRNNBlock 3 : 128 -> 256  + MaxPool(2,1)  -> (256,  3, 48)
        CRNNBlock 4 : 256 -> 256  (no pool)       -> (256,  3, 48)

    Map-to-Sequence:
        AdaptiveAvgPool2d((1, T))  collapses height -> (256, 1, T)
        Squeeze + Permute          -> (T, 256)

    Spatial Dropout:
        Dropout(0.3)  after CNN / before RNN

    RNN:
        BiLSTM  (input=256, hidden=256, layers=2, dropout=0.5)  -> (T, 512)

    Classifier:
        Linear(512, 37) -> (T, 37)   [36 chars + 1 CTC blank]

    Output:
        log-probabilities  (T, batch, 37)  ready for ``nn.CTCLoss``
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        img_h: int = IMG_SIZE,
        plate_w: int = PLATE_W,
        rnn_hidden: int = RNN_HIDDEN,
        rnn_layers: int = RNN_LAYERS,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            CRNNBlock(1,   64,  pool_kernel=(2, 2)),
            CRNNBlock(64,  128, pool_kernel=(2, 2)),
            CRNNBlock(128, 256, pool_kernel=(2, 1)),  # pool height only
            CRNNBlock(256, 256, pool_kernel=None),     # no pool
        )

        # Dynamically compute the CNN output shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, plate_w)
            cnn_out = self.cnn(dummy)
            _, C, H, W = cnn_out.shape

        # Collapse remaining height to 1 while preserving time-steps (width)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))

        # Spatial dropout between CNN and RNN for regularization
        self.cnn_dropout = nn.Dropout(0.3)

        self.rnn = nn.LSTM(
            input_size=C,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0,
        )

        self.fc = nn.Linear(rnn_hidden * 2, num_classes + 1)   # ×2 bidir

        self._T = W   # number of CTC time-steps
        assert self._T >= 2 * MAX_PLATE_CHARS + 1, (
            f"T={self._T} too small for MAX_PLATE_CHARS={MAX_PLATE_CHARS}"
        )
        log.info(
            "CRNN: CNN→(%d, %d, %d), T=%d time-steps, "
            "BiLSTM hidden=%d×2, output=%d classes",
            C, H, W, W, rnn_hidden, num_classes + 1,
        )

    @property
    def T(self) -> int:
        """Number of CTC time-steps produced by the CNN encoder."""
        return self._T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 1, H, W)

        Returns
        -------
        log_probs : (T, N, num_classes+1)
        """
        conv = self.cnn(x)                       # (N, C, H', W')
        conv = self.adaptive_pool(conv)           # (N, C, 1, T)
        conv = conv.squeeze(2)                    # (N, C, T)
        conv = conv.permute(0, 2, 1)              # (N, T, C)
        conv = self.cnn_dropout(conv)             # regularize CNN -> RNN

        rnn_out, _ = self.rnn(conv)               # (N, T, hidden*2)
        logits = self.fc(rnn_out)                 # (N, T, num_classes+1)

        log_probs = F.log_softmax(logits, dim=2)  # (N, T, C)
        log_probs = log_probs.permute(1, 0, 2)    # (T, N, C) — CTC format
        return log_probs


# ---------------------------------------------------------------------------
# 4. CTC Training Loop
# ---------------------------------------------------------------------------

class CRNNTrainer:
    """
    Training loop for CRNN with CTC loss.

    Mirrors ``Trainer`` in cnn_detection.py (same early-stopping, scheduler
    step, best-state restoration) but uses ``nn.CTCLoss`` instead of
    ``nn.CrossEntropyLoss``.

    Note: CTC does not support per-class weights or label smoothing;
    those CNN-specific options are not applicable here.
    """

    def __init__(
        self,
        model: CharCRNN,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        """Run one epoch; return average CTC loss."""
        self.model.train(train)
        total_loss = 0.0
        total_samples = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, targets, target_lengths in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                log_probs = self.model(images)        # (T, N, C)
                T, N, _ = log_probs.shape
                input_lengths = torch.full(
                    (N,), T, dtype=torch.long, device=self.device,
                )

                loss = self.ctc_loss(
                    log_probs, targets, input_lengths, target_lengths,
                )

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=5.0,
                    )
                    self.optimizer.step()

                total_loss += loss.item() * N
                total_samples += N

        return total_loss / max(total_samples, 1)

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = EPOCHS,
        early_stop_patience: int = EARLY_STOP_PATIENCE,
    ) -> None:
        """Train for up to *epochs* with early stopping and record history."""
        log.info(
            "Starting CRNN training on %s for up to %d epochs.",
            self.device, epochs,
        )
        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_count = 0

        for epoch in range(1, epochs + 1):
            tr_loss = self._run_epoch(train_loader, train=True)
            vl_loss = self._run_epoch(val_loader, train=False)

            self.train_losses.append(tr_loss)
            self.val_losses.append(vl_loss)

            if self.scheduler is not None:
                self.scheduler.step(vl_loss)

            if vl_loss < best_val_loss - 1e-4:
                best_val_loss = vl_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                patience_count = 0
            else:
                patience_count += 1

            log.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f "
                "[patience %d/%d]",
                epoch, epochs, tr_loss, vl_loss,
                patience_count, early_stop_patience,
            )

            if patience_count >= early_stop_patience:
                log.info(
                    "Early stopping triggered at epoch %d "
                    "(no val-loss improvement for %d epochs).",
                    epoch, early_stop_patience,
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            log.info("Restored best CRNN model (val_loss=%.4f).", best_val_loss)


# ---------------------------------------------------------------------------
# 5. CTC Decoding
# ---------------------------------------------------------------------------

def ctc_greedy_decode(
    log_probs: torch.Tensor,
    blank: int = BLANK_IDX,
) -> List[int]:
    """
    Greedy CTC decoding for a single sequence.

    Parameters
    ----------
    log_probs : (T, C) - log-probabilities for one sample
    blank     : blank token index

    Returns
    -------
    decoded : list of class indices (blanks and consecutive duplicates removed)
    """
    indices = log_probs.argmax(dim=1).cpu().tolist()     # (T,)
    decoded: List[int] = []
    prev = -1
    for idx in indices:
        if idx != blank and idx != prev:
            decoded.append(idx)
        prev = idx
    return decoded


def ctc_greedy_decode_batch(
    log_probs: torch.Tensor,
    blank: int = BLANK_IDX,
) -> List[List[int]]:
    """Batch-wise greedy CTC decoding.  ``log_probs``: (T, N, C)."""
    _, N, _ = log_probs.shape
    return [ctc_greedy_decode(log_probs[:, i, :], blank) for i in range(N)]


def ctc_beam_decode(
    log_probs: torch.Tensor,
    beam_width: int = BEAM_WIDTH,
    blank: int = BLANK_IDX,
) -> List[int]:
    """
    Prefix beam search CTC decoding for a single sequence.

    Maintains separate probability scores for beams ending in blank vs
    non-blank, which is essential for correctly handling repeated characters
    (e.g., "AA" requires a blank between the two A emissions).

    Parameters
    ----------
    log_probs : (T, C) - log-probabilities for one sample
    beam_width : number of beams to maintain
    blank      : blank token index

    Returns
    -------
    decoded : list of class indices (best beam after collapsing)
    """
    T, C = log_probs.shape
    probs = log_probs.exp().cpu().numpy()   # (T, C) probabilities

    # Each beam tracks: prefix -> (p_blank, p_non_blank)
    # p_blank     = probability of all paths that end with blank and collapse to prefix
    # p_non_blank = probability of all paths that end with a non-blank char
    beams: dict[tuple, list] = {(): [1.0, 0.0]}   # (p_blank, p_non_blank)

    for t in range(T):
        new_beams: dict[tuple, list] = {}

        def _add(prefix: tuple, p_b: float, p_nb: float) -> None:
            if prefix in new_beams:
                new_beams[prefix][0] += p_b
                new_beams[prefix][1] += p_nb
            else:
                new_beams[prefix] = [p_b, p_nb]

        for prefix, (p_b, p_nb) in beams.items():
            p_total = p_b + p_nb

            # --- emit blank: prefix stays the same, score goes to p_blank ---
            _add(prefix, p_total * float(probs[t, blank]), 0.0)

            # --- emit each non-blank character ---
            for c in range(C):
                if c == blank:
                    continue
                p_c = float(probs[t, c])

                if prefix and c == prefix[-1]:
                    # Same char as the end of prefix:
                    #   - paths that ended with blank CAN extend (produces repeated char)
                    #   - paths that ended with non-blank just continue (collapse)
                    _add(prefix + (c,), 0.0, p_b * p_c)    # blank -> c = new char
                    _add(prefix, 0.0, p_nb * p_c)           # non-blank -> c = collapse
                else:
                    # Different char: always extends the prefix
                    _add(prefix + (c,), 0.0, p_total * p_c)

        # Prune to top beam_width beams by total probability
        sorted_beams = sorted(
            new_beams.items(),
            key=lambda x: x[1][0] + x[1][1],
            reverse=True,
        )
        beams = {k: v for k, v in sorted_beams[:beam_width]}

    # Return the best beam
    if not beams:
        return []
    best_prefix = max(beams, key=lambda k: beams[k][0] + beams[k][1])
    return list(best_prefix)


# ---------------------------------------------------------------------------
# 6. CRNN Plate Prediction
# ---------------------------------------------------------------------------

def preprocess_plate_strip(
    bgr: np.ndarray,
    img_h: int = IMG_SIZE,
    plate_w: int = PLATE_W,
) -> np.ndarray:
    """Convert a BGR plate image into a fixed-size grayscale strip (H, W) with CLAHE."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE contrast normalization (same adaptive logic as dataset loader)
    std = float(np.std(gray))
    clip = 4.0 if std < 30 else (1.0 if std > 80 else 2.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    h, w = gray.shape
    new_w = max(1, int(w * img_h / h))
    resized = cv2.resize(gray, (new_w, img_h), interpolation=cv2.INTER_AREA)

    if new_w < plate_w:
        padded = np.zeros((img_h, plate_w), dtype=np.uint8)
        padded[:, :new_w] = resized
        return padded
    else:
        return cv2.resize(resized, (plate_w, img_h),
                          interpolation=cv2.INTER_AREA)


def crnn_predict_plate(
    bgr: np.ndarray,
    model: CharCRNN,
    device: torch.device,
    plate_w: int = PLATE_W,
) -> str:
    """
    Predict the full plate text from a raw plate image using CRNN + CTC.

    Unlike the CNN pipeline this does **not** require character segmentation;
    the entire plate strip is fed through the network and CTC decoding
    produces the text directly.
    """
    img = preprocess_plate_strip(bgr, IMG_SIZE, plate_w)

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    tensor = tf(img).unsqueeze(0).to(device)             # (1, 1, H, W)

    model.eval()
    with torch.no_grad():
        log_probs = model(tensor)                        # (T, 1, C)

    decoded = ctc_beam_decode(log_probs[:, 0, :], beam_width=BEAM_WIDTH)

    if not decoded:
        return "<no_chars>"
    return "".join(idx_to_char(idx) for idx in decoded)


# ---------------------------------------------------------------------------
# 7. Character-level Evaluation for CRNN
# ---------------------------------------------------------------------------

def evaluate_crnn_characters(
    model: CharCRNN,
    test_paths: List[Path],
    test_labels: List[str],
    device: torch.device,
    plate_w: int = PLATE_W,
) -> dict:
    """
    Character-level evaluation by running CRNN on held-out test plates.

    CTC-decoded sequences are padded / truncated to the ground-truth label
    length so that position-wise  (predicted_class, true_class) pairs can be
    collected and fed to the same ``sklearn`` metric functions used by the CNN
    ``Evaluator``.
    """
    all_pred_indices: List[int] = []
    all_true_indices: List[int] = []

    model.eval()
    for path, label in zip(test_paths, test_labels):
        bgr = cv2.imread(str(path))
        if bgr is None:
            # count every GT char as a miss
            for ch in label:
                t_idx = char_to_idx(ch)
                if t_idx is not None:
                    all_true_indices.append(t_idx)
                    all_pred_indices.append((t_idx + 1) % NUM_CLASSES)
            continue

        predicted = crnn_predict_plate(bgr, model, device, plate_w)
        if predicted == "<no_chars>":
            predicted = ""

        gt_len = len(label)
        pred_chars = list(predicted)

        # Truncate if CRNN produced more characters than label
        if len(pred_chars) > gt_len:
            pred_chars = pred_chars[:gt_len]

        # Position-wise comparison
        for i in range(gt_len):
            t_idx = char_to_idx(label[i])
            if t_idx is None:
                continue
            if i < len(pred_chars):
                p_idx = char_to_idx(pred_chars[i])
                if p_idx is None:
                    p_idx = (t_idx + 1) % NUM_CLASSES     # unknown → wrong
            else:
                # Predicted too short → count as wrong for this char
                p_idx = (t_idx + 1) % NUM_CLASSES

            all_pred_indices.append(p_idx)
            all_true_indices.append(t_idx)

    if not all_true_indices:
        return {
            "accuracy": 0.0, "precision": 0.0,
            "recall": 0.0, "f1": 0.0,
        }

    acc  = accuracy_score(all_true_indices, all_pred_indices)
    prec = precision_score(all_true_indices, all_pred_indices,
                           average="weighted", zero_division=0)
    rec  = recall_score(all_true_indices, all_pred_indices,
                        average="weighted", zero_division=0)
    f1   = f1_score(all_true_indices, all_pred_indices,
                    average="weighted", zero_division=0)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # Per-class F1 for confusable pairs (same chars as CNN)
    per_class_f1 = f1_score(
        all_true_indices, all_pred_indices,
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    for ch in ["B", "8", "O", "0", "S", "5", "I", "1", "Q", "Z", "2"]:
        idx = char_to_idx(ch)
        if idx is not None and idx < len(per_class_f1):
            metrics[f"f1_{ch}"] = float(per_class_f1[idx])

    # Pretty-print (matches CNN Evaluator output style)
    print("\n" + "=" * 60)
    print("CRNN  CHARACTER-LEVEL  EVALUATION  (test plates)")
    print("=" * 60)
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%  (weighted)")
    print(f"  Recall    : {rec  * 100:.2f}%  (weighted)")
    print(f"  F1-Score  : {f1   * 100:.2f}%  (weighted)")
    print("=" * 60)

    present_classes = sorted(set(all_true_indices) | set(all_pred_indices))
    valid_classes   = [c for c in present_classes if c < NUM_CLASSES]
    target_names    = [idx_to_char(i) for i in valid_classes]
    print("\nPer-class report:")
    print(
        classification_report(
            all_true_indices, all_pred_indices,
            labels=valid_classes,
            target_names=target_names,
            zero_division=0,
        )
    )
    return metrics


# ---------------------------------------------------------------------------
# 8. Plate-level Evaluation (all plates — same scope as CNN)
# ---------------------------------------------------------------------------

def evaluate_crnn_plates(
    dataset_root: Path,
    model: CharCRNN,
    device: torch.device,
    plate_w: int = PLATE_W,
    img_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> dict:
    """
    Run plate-level inference over **every image** in the dataset (train +
    val + test) and compute exact-match accuracy, mean CER, and mean
    inference time — identical evaluation scope as ``evaluate_plates`` in
    cnn_detection.py.
    """
    all_paths: List[Path] = []
    for sub in ("train", "val", "test"):
        folder = dataset_root / sub
        if folder.exists():
            all_paths.extend(
                p for p in folder.iterdir()
                if p.suffix.lower() in img_extensions
            )

    if not all_paths:
        log.warning("evaluate_crnn_plates: no images found.")
        return {
            "exact_match_acc": 0.0, "mean_cer": 1.0,
            "mean_inference_ms": 0.0, "n_plates": 0,
        }

    exact_matches = 0
    cer_values: List[float] = []
    infer_times: List[float] = []

    log.info("CRNN plate-level evaluation on %d plates …", len(all_paths))
    for img_path in all_paths:
        actual = extract_label_from_filename(img_path)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            cer_values.append(1.0)
            continue

        t0 = time.perf_counter()
        predicted = crnn_predict_plate(bgr, model, device, plate_w)
        t1 = time.perf_counter()
        infer_times.append((t1 - t0) * 1000.0)

        if predicted in ("<no_chars>", "<read_error>"):
            cer_values.append(1.0)
            continue

        ed  = edit_distance(predicted, actual)
        cer = ed / max(len(actual), 1)
        cer_values.append(min(cer, 1.0))

        if predicted == actual:
            exact_matches += 1

    n = len(cer_values)
    return {
        "exact_match_acc":   exact_matches / max(n, 1),
        "mean_cer":          float(np.mean(cer_values))  if cer_values  else 1.0,
        "mean_inference_ms": float(np.mean(infer_times)) if infer_times else 0.0,
        "n_plates":          len(all_paths),
    }


# ---------------------------------------------------------------------------
# 9. Plate Prediction Demo
# ---------------------------------------------------------------------------

def predict_plates_demo(
    dataset_root: Path,
    model: CharCRNN,
    device: torch.device,
    plate_w: int = PLATE_W,
    n_plates: int = 20,
    img_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> None:
    """
    Run CRNN on ``n_plates`` random plates (same sample as CNN demo thanks
    to ``random.Random(SEED)``) and print a side-by-side comparison.
    """
    all_paths: List[Path] = []
    for sub in ("train", "val", "test"):
        folder = dataset_root / sub
        if folder.exists():
            all_paths.extend(
                p for p in folder.iterdir()
                if p.suffix.lower() in img_extensions
            )

    if not all_paths:
        log.warning("No plates found for demo.")
        return

    rng = random.Random(SEED)
    sample = rng.sample(all_paths, min(n_plates, len(all_paths)))

    print("\n" + "=" * 62)
    print(f"  CRNN PLATE PREDICTION DEMO  ({len(sample)} plates)")
    print("=" * 62)
    print(f"  {'#':>3}  {'Actual':<12}  {'Predicted':<12}  {'Match?'}")
    print("  " + "-" * 58)

    correct = 0
    for i, img_path in enumerate(sample, start=1):
        actual = extract_label_from_filename(img_path)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            predicted = "<read_error>"
        else:
            predicted = crnn_predict_plate(bgr, model, device, plate_w)

        match = "Y" if predicted == actual else "X"  # checkmark vs cross
        if predicted == actual:
            correct += 1
        print(f"  {i:>3}  {actual:<12}  {predicted:<12}  {match}")

    print("  " + "-" * 58)
    print(
        f"  Plate-level accuracy: {correct}/{len(sample)} "
        f"({correct / len(sample) * 100:.1f}%)"
    )
    print("=" * 62)


# ---------------------------------------------------------------------------
# 10. Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end CRNN pipeline: download → train → evaluate → export."""

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    base_path    = download_dataset()
    dataset_root = find_dataset_root(base_path)
    log.info("Using dataset root: %s", dataset_root)

    # ── 2. Character segmentation stats (same as CNN — shared seg_stats) ───
    segmenter = CharacterSegmenter(
        img_size=IMG_SIZE,
        min_height_ratio=MIN_CHAR_HEIGHT_RATIO,
        min_area=MIN_CHAR_AREA,
        max_width_ratio=MAX_CHAR_WIDTH_RATIO,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
        min_char_aspect=MIN_CHAR_ASPECT,
        max_char_aspect=MAX_CHAR_ASPECT,
        vert_center_tol=VERT_CENTER_TOL,
        border_exclusion_ratio=BORDER_EXCLUSION_RATIO,
        min_width_ratio_vs_median=MIN_WIDTH_RATIO_VS_MEDIAN,
        min_height_ratio_vs_median=MIN_HEIGHT_RATIO_VS_MEDIAN,
        max_height_ratio_vs_median=MAX_HEIGHT_RATIO_VS_MEDIAN,
    )
    _, _, seg_stats = build_char_dataset(dataset_root, segmenter)

    # ── 3. Build plate-level dataset ───────────────────────────────────────
    plate_paths, plate_labels = build_plate_dataset(dataset_root)

    # Drop plates whose label-length group has fewer than 2 members
    # (cannot be stratified, mirrors CNN's rare-class pruning)
    len_counts = Counter(len(l) for l in plate_labels)
    keep_mask = [len_counts[len(l)] >= 2 for l in plate_labels]
    removed = sum(1 for k in keep_mask if not k)
    if removed:
        log.warning(
            "Dropped %d plates with rare label lengths (cannot stratify).",
            removed,
        )
    plate_paths  = [p for p, k in zip(plate_paths, keep_mask) if k]
    plate_labels = [l for l, k in zip(plate_labels, keep_mask) if k]

    # ── 4. Train / test split (80 : 20) ───────────────────────────────────
    try:
        paths_train, paths_test, labels_train, labels_test = train_test_split(
            plate_paths, plate_labels,
            test_size=0.20, random_state=SEED,
            stratify=[len(l) for l in plate_labels],
        )
    except ValueError as exc:
        log.warning("Stratified plate split failed (%s); fallback.", exc)
        paths_train, paths_test, labels_train, labels_test = train_test_split(
            plate_paths, plate_labels,
            test_size=0.20, random_state=SEED,
        )
    log.info(
        "Plate split | train=%d  test=%d",
        len(labels_train), len(labels_test),
    )

    # ── 5. DataLoaders ─────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_crnn_dataloaders(
        paths_train, labels_train,
        paths_test,  labels_test,
        batch_size=BATCH_SIZE,
    )

    # ── 6. Model ───────────────────────────────────────────────────────────
    device = get_device()
    model = CharCRNN(
        num_classes=NUM_CLASSES,
        img_h=IMG_SIZE,
        plate_w=PLATE_W,
        rnn_hidden=RNN_HIDDEN,
        rnn_layers=RNN_LAYERS,
        dropout=DROPOUT_RATE,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    log.info("CRNN trainable parameters: %d", total_params)

    # -- 7. Optimizer & scheduler -------------------------------------------
    optimizer = optim.Adam(
        model.parameters(), lr=CRNN_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # -- 8. Training --------------------------------------------------------
    trainer = CRNNTrainer(model, optimizer, device, scheduler)
    trainer.fit(train_loader, val_loader, epochs=CRNN_EPOCHS)

    # ── 9. Save model weights ──────────────────────────────────────────────
    weights_path = Path("char_crnn_weights.pth")
    torch.save(model.state_dict(), weights_path)
    log.info("CRNN weights saved → %s", weights_path)

    # ── 10. Character-level evaluation (test plates) ───────────────────────
    char_metrics = evaluate_crnn_characters(
        model, paths_test, labels_test, device, PLATE_W,
    )

    # ── 11. Loss curves ───────────────────────────────────────────────────
    curve_path = Path("loss_curves_crnn.png")
    plot_loss_curves(
        trainer.train_losses,
        trainer.val_losses,
        save_path=curve_path,
    )

    # ── 12. Plate prediction demo (20 plates, same seed) ──────────────────
    predict_plates_demo(dataset_root, model, device, PLATE_W, n_plates=20)

    # ── 13. Full plate-level evaluation & CSV export ──────────────────────
    plate_metrics = evaluate_crnn_plates(dataset_root, model, device, PLATE_W)

    all_metrics: dict = {
        # ── Identity ─────────────────────────────────────────────────────
        "model_type":               "CRNN",
        "model_trainable_params":   total_params,
        # ── Character-level (test plates) ────────────────────────────────
        "char_accuracy":            round(char_metrics["accuracy"],  4),
        "char_precision_weighted":  round(char_metrics["precision"], 4),
        "char_recall_weighted":     round(char_metrics["recall"],    4),
        "char_f1_weighted":         round(char_metrics["f1"],        4),
        # Per-class F1 for visually confusable pairs
        **{
            f"char_f1_{ch}": round(char_metrics.get(f"f1_{ch}", 0.0), 4)
            for ch in ["B", "8", "O", "0", "S", "5", "I", "1", "Q", "Z", "2"]
        },
        # ── Plate-level (all plates) ─────────────────────────────────────
        "plate_exact_match_acc":    round(plate_metrics["exact_match_acc"],   4),
        "plate_mean_cer":           round(plate_metrics["mean_cer"],          4),
        "plate_mean_inference_ms":  round(plate_metrics["mean_inference_ms"], 2),
        "plate_n_evaluated":        plate_metrics["n_plates"],
        # ── Segmentation (from build_char_dataset — identical to CNN) ────
        "seg_total_plates":         seg_stats["total_plates"],
        "seg_successful_plates":    seg_stats["successful_plates"],
        "seg_success_rate":         round(seg_stats["seg_success_rate"], 4),
    }

    save_metrics_csv(all_metrics, Path("crnn_metrics.csv"))
    log.info("Done.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
