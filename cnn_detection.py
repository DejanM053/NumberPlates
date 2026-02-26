"""
European License Plate Recognition System
==========================================
A CNN-based character classifier for European license plates.

Dataset  : https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset
           Folder layout after download:
               dataset_final/
                   train/   (80 %)  – plate images, filename = plate text
                   val/     (10 %)
                   test/    (10 %)

Pipeline :
    1. Download dataset via kagglehub
    2. Collect all plate images from train + val + test folders
    3. Segment individual characters from each plate image
       (grayscale → CLAHE → Otsu threshold → contour detection)
    4. Match segmented blobs L→R to the label string in the filename
    5. Build a flat character dataset  (image_array, class_index)
    6. Re-split  80 % train / 20 % test  (character level)
    7. Train a CNN character classifier (PyTorch)
    8. Evaluate: accuracy, precision, recall, F1
    9. Plot training / validation loss curves
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import csv
import logging
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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
# Global configuration
# ---------------------------------------------------------------------------
IMG_SIZE: int = 28               # character image size (pixels) fed to the CNN
NUM_CLASSES: int = 36            # 0–9 (10) + A–Z (26)
BATCH_SIZE: int = 64
EPOCHS: int = 200                 # more epochs; early stopping will cut training short
LEARNING_RATE: float = 1e-3
DROPOUT_RATE: float = 0.5        # increased from 0.4 → stronger regularisation
WEIGHT_DECAY: float = 5e-4       # increased from 1e-4 → penalise large weights more
LABEL_SMOOTHING: float = 0.10    # prevent overconfident predictions
EARLY_STOP_PATIENCE: int = 10    # stop after N epochs without val-loss improvement
SEED: int = 2003
MIN_CHAR_HEIGHT_RATIO: float = 0.20   # contour must span at least 20 % of plate height
MIN_CHAR_AREA: int = 50               # minimum contour area (pixels²)
MAX_CHAR_WIDTH_RATIO: float = 0.25    # contour must not exceed 25 % of plate width
NMS_IOU_THRESHOLD: float = 0.30       # suppress overlapping boxes above this IoU
MIN_CHAR_ASPECT: float = 0.20         # min w/h ratio  (tightened from 0.15)
MAX_CHAR_ASPECT: float = 0.90         # max w/h ratio  (tightened from 1.20)
VERT_CENTER_TOL: float = 0.28         # max deviation of box v-center from median (fraction of plate height)
MIN_PLATE_CHARS: int = 4              # shortest valid European plate text
MAX_PLATE_CHARS: int = 10             # longest valid European plate text
TYPICAL_PLATE_LEN: int = 7            # used as fallback when expected_count is unknown
# Segmentation consistency filters (Fix 1a–1d)
BORDER_EXCLUSION_RATIO: float = 0.08  # exclude boxes whose edge falls within this fraction of plate edge
MIN_WIDTH_RATIO_VS_MEDIAN: float = 0.35   # drop box if width < this × median width
MIN_HEIGHT_RATIO_VS_MEDIAN: float = 0.55  # drop box if height < this × median height
MAX_HEIGHT_RATIO_VS_MEDIAN: float = 1.45  # drop box if height > this × median height
# Partial-match dataset collection (Fix 5a–5b)
ALLOW_PARTIAL_MATCH: bool = True      # allow ±1 box tolerance when building char dataset
# Two-pass tightening at inference time (Fix 2b)
MAX_TIGHTEN_ATTEMPTS: int = 5         # max refinement passes in predict_plate()

# Reproducibility
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
# Character ↔ class-index helpers
# ---------------------------------------------------------------------------
#  Classes: '0'–'9' → 0–9,  'A'–'Z' → 10–35
VALID_CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX: dict[str, int] = {ch: i for i, ch in enumerate(VALID_CHARS)}
IDX_TO_CHAR: dict[int, str] = {i: ch for ch, i in CHAR_TO_IDX.items()}


def char_to_idx(ch: str) -> int | None:
    """Return class index for a character, or None if not alphanumeric."""
    return CHAR_TO_IDX.get(ch.upper())


def idx_to_char(idx: int) -> str:
    """Return the character for a class index."""
    return IDX_TO_CHAR[idx]


def edit_distance(a: str, b: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.
    Used to calculate Character Error Rate (CER) at plate level.
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# 1. Character Segmentation
# ---------------------------------------------------------------------------

class CharacterSegmenter:
    """
    Extracts individual character images from a cropped license plate image.

    Strategy
    --------
    Four preprocessing strategies are tried per plate:
      1. Otsu inverted  (dark chars on light background)
      2. Otsu normal    (light chars on dark background)
      3. Adaptive Gaussian thresholding
      4. Bilateral filter + Otsu inverted (low-contrast / dark plates)

    The strategy whose box count is *closest* to expected_count (if known),
    or closest to TYPICAL_PLATE_LEN within [MIN_PLATE_CHARS, MAX_PLATE_CHARS]
    otherwise, is selected.  If no strategy falls inside the valid window,
    the globally best-scoring strategy is used.

    Filtering pipeline (applied per strategy candidate set)
    --------------------------------------------------------
    1.  Grayscale + adaptive-CLAHE contrast normalisation (Fix 4b).
    2.  Thresholding (four variants, Fix 4a).
    3.  Height filter    : h ≥ min_height_ratio × plate height.
    4.  Area filter      : contour area ≥ min_area px².
    5.  Width filter     : w ≤ max_width_ratio × plate width.
    6.  Aspect filter    : min_char_aspect ≤ w/h ≤ max_char_aspect  (Fix 1c).
    7.  NMS              : suppress heavily overlapping boxes.
    8.  Vertical-center filter : remove vertical outliers  (existing).
    9.  Horizontal zone  : remove boxes fully inside left/right border band (Fix 1a).
    10. Width consistency: remove boxes narrower than MIN_WIDTH_RATIO_VS_MEDIAN
                           × median width  (Fix 1b).
    11. Height consistency: remove boxes outside [MIN…MAX]_HEIGHT_RATIO_VS_MEDIAN
                            × median height  (Fix 1d).
    12. Sort left → right.
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        min_height_ratio: float = MIN_CHAR_HEIGHT_RATIO,
        min_area: int = MIN_CHAR_AREA,
        max_width_ratio: float = MAX_CHAR_WIDTH_RATIO,
        nms_iou_threshold: float = NMS_IOU_THRESHOLD,
        min_char_aspect: float = MIN_CHAR_ASPECT,
        max_char_aspect: float = MAX_CHAR_ASPECT,
        vert_center_tol: float = VERT_CENTER_TOL,
        border_exclusion_ratio: float = BORDER_EXCLUSION_RATIO,
        min_width_ratio_vs_median: float = MIN_WIDTH_RATIO_VS_MEDIAN,
        min_height_ratio_vs_median: float = MIN_HEIGHT_RATIO_VS_MEDIAN,
        max_height_ratio_vs_median: float = MAX_HEIGHT_RATIO_VS_MEDIAN,
    ) -> None:
        self.img_size                  = img_size
        self.min_height_ratio          = min_height_ratio
        self.min_area                  = min_area
        self.max_width_ratio           = max_width_ratio
        self.nms_iou_threshold         = nms_iou_threshold
        self.min_char_aspect           = min_char_aspect
        self.max_char_aspect           = max_char_aspect
        self.vert_center_tol           = vert_center_tol
        self.border_exclusion_ratio    = border_exclusion_ratio
        self.min_width_ratio_vs_median = min_width_ratio_vs_median
        self.min_height_ratio_vs_median = min_height_ratio_vs_median
        self.max_height_ratio_vs_median = max_height_ratio_vs_median

    # ------------------------------------------------------------------
    @staticmethod
    def _adaptive_clahe(gray: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE with a clip limit that adapts to the image's global
        contrast (Fix 4b).

        Low contrast  (std < 30) → clipLimit 4.0 (aggressive enhancement)
        High contrast (std > 80) → clipLimit 1.0 (light touch)
        Otherwise                → clipLimit 2.0
        """
        std = float(np.std(gray))
        if std < 30:
            clip = 4.0
        elif std > 80:
            clip = 1.0
        else:
            clip = 2.0
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _clean(binary: np.ndarray) -> np.ndarray:
        """Remove salt-and-pepper noise with a small morphological open."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def _candidate_binaries(self, bgr: np.ndarray) -> List[np.ndarray]:
        """
        Return four binary images from different preprocessing strategies.

        Strategy 1: adaptive-CLAHE gray + Otsu inverted
        Strategy 2: adaptive-CLAHE gray + Otsu normal
        Strategy 3: adaptive-CLAHE gray + Adaptive Gaussian
        Strategy 4: bilateral filter   + Otsu inverted  (Fix 4a – low-contrast fallback)
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_clahe = self._adaptive_clahe(gray)
        candidates: List[np.ndarray] = []

        # Strategy 1: dark chars on light plate
        _, b1 = cv2.threshold(gray_clahe, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        candidates.append(self._clean(b1))

        # Strategy 2: light chars on dark plate
        _, b2 = cv2.threshold(gray_clahe, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(self._clean(b2))

        # Strategy 3: adaptive Gaussian – handles uneven lighting
        b3 = cv2.adaptiveThreshold(
            gray_clahe, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            blockSize=15, C=8,
        )
        candidates.append(self._clean(b3))

        # Strategy 4 (Fix 4a): bilateral filter preserves char edges on
        # low-contrast / dark plates where CLAHE is not enough
        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        _, b4 = cv2.threshold(bilateral, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        candidates.append(self._clean(b4))

        return candidates

    # ------------------------------------------------------------------
    @staticmethod
    def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
        """Compute intersection-over-union for two (x,y,w,h) boxes."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, boxes: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
        """Remove boxes that overlap heavily with a larger-area box."""
        if len(boxes) <= 1:
            return boxes
        sorted_b = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        keep: List[Tuple[int,int,int,int]] = []
        suppressed = [False] * len(sorted_b)
        for i, box_i in enumerate(sorted_b):
            if suppressed[i]:
                continue
            keep.append(box_i)
            for j in range(i + 1, len(sorted_b)):
                if not suppressed[j]:
                    if self._iou(box_i, sorted_b[j]) > self.nms_iou_threshold:
                        suppressed[j] = True
        return keep

    # ------------------------------------------------------------------
    def _filter_vertical_outliers(
        self,
        boxes: List[Tuple[int, int, int, int]],
        plate_height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Remove boxes whose vertical centre strays too far from the median."""
        if len(boxes) <= 2:
            return boxes
        centers = np.array([y + h / 2 for _, y, _, h in boxes])
        median_c = float(np.median(centers))
        tol = plate_height * self.vert_center_tol
        return [b for b, c in zip(boxes, centers) if abs(c - median_c) <= tol]

    def _filter_border_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        plate_width: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Fix 1a – Remove boxes that sit entirely within the border exclusion band.

        Any box whose RIGHT edge is within the leftmost (border_exclusion_ratio)
        fraction of the plate, or whose LEFT edge is within the rightmost
        fraction, is considered a frame / EU-strip artefact and removed.
        """
        if not boxes:
            return boxes
        left_band  = plate_width * self.border_exclusion_ratio
        right_band = plate_width * (1.0 - self.border_exclusion_ratio)
        return [
            (x, y, w, h) for x, y, w, h in boxes
            if not ((x + w) <= left_band or x >= right_band)
        ]

    def _filter_width_consistency(
        self,
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Fix 1b – Remove boxes that are much narrower than the median box width.

        Bolt holes and EU-strip fragments are usually narrow slivers;
        real alphanumeric characters are comparably wide.
        """
        if len(boxes) <= 2:
            return boxes
        median_w = float(np.median([w for _, _, w, _ in boxes]))
        min_w = self.min_width_ratio_vs_median * median_w
        return [(x, y, w, h) for x, y, w, h in boxes if w >= min_w]

    def _filter_height_consistency(
        self,
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Fix 1d – Remove boxes whose height is an outlier relative to the median.

        Keeps only boxes within [MIN_HEIGHT_RATIO_VS_MEDIAN, MAX_HEIGHT_RATIO_VS_MEDIAN]
        × the median height.
        """
        if len(boxes) <= 2:
            return boxes
        median_h = float(np.median([h for _, _, _, h in boxes]))
        lo = self.min_height_ratio_vs_median * median_h
        hi = self.max_height_ratio_vs_median * median_h
        return [(x, y, w, h) for x, y, w, h in boxes if lo <= h <= hi]

    # ------------------------------------------------------------------
    def _find_char_boxes(
        self,
        binary: np.ndarray,
        apply_consistency_filters: bool = True,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find, filter, and return (x,y,w,h) character boxes sorted left→right.

        Parameters
        ----------
        binary                    : thresholded plate image
        apply_consistency_filters : when True, applies Fix 1a/1b/1d filters
                                    on top of the primary filters.
        """
        h_img, w_img = binary.shape[:2]
        min_h = int(h_img * self.min_height_ratio)
        max_w = int(w_img * self.max_width_ratio)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if h == 0:
                continue
            aspect = w / h
            if (
                h >= min_h
                and area >= self.min_area
                and w <= max_w
                and self.min_char_aspect <= aspect <= self.max_char_aspect
            ):
                boxes.append((x, y, w, h))

        boxes = self._nms(boxes)
        boxes = self._filter_vertical_outliers(boxes, h_img)

        if apply_consistency_filters:
            boxes = self._filter_border_boxes(boxes, w_img)
            boxes = self._filter_width_consistency(boxes)
            boxes = self._filter_height_consistency(boxes)

        boxes.sort(key=lambda b: b[0])   # left → right
        return boxes

    # ------------------------------------------------------------------
    def segment(
        self,
        bgr: np.ndarray,
        expected_count: int | None = None,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Segment characters from a plate image.

        Parameters
        ----------
        bgr : np.ndarray
            The cropped plate image in BGR colour space.
        expected_count : int, optional
            Expected number of characters (from filename label).
            When provided, the strategy whose box count is closest to this
            value is selected.  When None, the strategy closest to
            TYPICAL_PLATE_LEN within [MIN_PLATE_CHARS, MAX_PLATE_CHARS] is
            preferred (Fix 2a).

        Returns
        -------
        char_imgs : list of np.ndarray  – (img_size × img_size) uint8 patches
        boxes     : list of (x, y, w, h)
        """
        binaries = self._candidate_binaries(bgr)
        all_boxes = [self._find_char_boxes(b) for b in binaries]

        if expected_count is not None:
            diffs = [abs(len(b) - expected_count) for b in all_boxes]
            best_idx = int(np.argmin(diffs))
        else:
            # Fix 2a: prefer strategies whose count is in the valid window;
            # within that window prefer closest to TYPICAL_PLATE_LEN.
            def _score(n: int) -> int:
                in_range = MIN_PLATE_CHARS <= n <= MAX_PLATE_CHARS
                return abs(n - TYPICAL_PLATE_LEN) + (0 if in_range else 1000)

            best_idx = int(min(range(len(all_boxes)),
                               key=lambda i: _score(len(all_boxes[i]))))

            # If even the best is still outside the window, try re-running
            # _find_char_boxes WITHOUT the consistency filters to see whether
            # that gets us into range (Fix 2a – hard window enforcement).
            if not (MIN_PLATE_CHARS <= len(all_boxes[best_idx]) <= MAX_PLATE_CHARS):
                relaxed_boxes = [self._find_char_boxes(b, apply_consistency_filters=False)
                                 for b in binaries]
                r_best = int(min(range(len(relaxed_boxes)),
                                 key=lambda i: _score(len(relaxed_boxes[i]))))
                if _score(len(relaxed_boxes[r_best])) < _score(len(all_boxes[best_idx])):
                    all_boxes = relaxed_boxes
                    best_idx  = r_best

        binary = binaries[best_idx]
        boxes  = all_boxes[best_idx]

        char_imgs: List[np.ndarray] = []
        for x, y, w, h in boxes:
            pad = max(2, int(0.05 * max(w, h)))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(binary.shape[1], x + w + pad)
            y2 = min(binary.shape[0], y + h + pad)

            patch   = binary[y1:y2, x1:x2]
            resized = cv2.resize(patch, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_AREA)
            char_imgs.append(resized)

        return char_imgs, boxes


# ---------------------------------------------------------------------------
# 2. Build the flat character dataset from raw plate images
# ---------------------------------------------------------------------------

def extract_label_from_filename(filepath: Path) -> str:
    """
    Strip the plate label from the image filename.

    The dataset stores filenames like  'ABC1234.jpg'  or  'AB-12-CD.jpg'.
    We keep only alphanumeric characters and convert to upper-case.
    """
    stem = filepath.stem
    return re.sub(r"[^A-Za-z0-9]", "", stem).upper()


def build_char_dataset(
    dataset_root: Path,
    segmenter: CharacterSegmenter,
    subfolders: Tuple[str, ...] = ("train", "val", "test"),
    img_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk every plate image, segment characters, and build a labelled array.

    Parameters
    ----------
    dataset_root : Path
        Root directory that contains the sub-folders (train / val / test).
    segmenter : CharacterSegmenter
    subfolders : tuple of str
        Sub-folder names to scan.
    img_extensions : tuple of str
        Accepted image extensions.

    Returns
    -------
    X : np.ndarray, shape (N, img_size, img_size)  – uint8 grayscale patches
    y : np.ndarray, shape (N,)                      – integer class labels
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    skipped_plates = 0
    total_plates = 0

    for subfolder in subfolders:
        folder = dataset_root / subfolder
        if not folder.exists():
            log.warning("Subfolder not found, skipping: %s", folder)
            continue

        image_paths = [
            p for p in folder.iterdir()
            if p.suffix.lower() in img_extensions
        ]
        log.info("Scanning %s: %d images", folder.name, len(image_paths))

        for img_path in image_paths:
            total_plates += 1
            label_str = extract_label_from_filename(img_path)

            # Only keep known alphanumeric characters
            label_chars = [ch for ch in label_str if char_to_idx(ch) is not None]
            if not label_chars:
                skipped_plates += 1
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                skipped_plates += 1
                continue

            # Pass expected count so the segmenter picks the best strategy
            char_imgs, _ = segmenter.segment(bgr, expected_count=len(label_chars))

            # ── Match segmented blobs to label characters ─────────────────
            n_seg   = len(char_imgs)
            n_label = len(label_chars)

            if n_seg == n_label:
                # Exact match – normal path
                pass

            elif n_seg > n_label:
                # Fix 5a – try trimming border artefacts (one char too many)
                plate_w = bgr.shape[1]
                border  = plate_w * BORDER_EXCLUSION_RATIO
                # Reload segment with boxes so we can inspect x-positions
                _, boxes_now = segmenter.segment(bgr, expected_count=n_label)
                trimmed_imgs = char_imgs
                if len(boxes_now) == n_seg:
                    if n_seg == n_label + 1:
                        # Try dropping the first box if it is a border artefact
                        if boxes_now[0][0] + boxes_now[0][2] <= border:
                            char_imgs  = char_imgs[1:]
                            n_seg     -= 1
                        # Try dropping the last box if it is a border artefact
                        elif boxes_now[-1][0] >= plate_w - border:
                            char_imgs  = char_imgs[:-1]
                            n_seg     -= 1

                # Fix 5b – if still one too many and ALLOW_PARTIAL_MATCH,
                # remove each box one at a time, keep the removal that
                # produces the most uniform horizontal spacing.
                if n_seg == n_label + 1 and ALLOW_PARTIAL_MATCH and len(char_imgs) >= 2:
                    _, boxes_now = segmenter.segment(bgr, expected_count=n_label)
                    if len(boxes_now) == n_seg:
                        best_std   = float("inf")
                        best_imgs  = None
                        for drop_i in range(n_seg):
                            remaining_boxes = [
                                b for j, b in enumerate(boxes_now) if j != drop_i
                            ]
                            if len(remaining_boxes) >= 2:
                                x_centers = [b[0] + b[2] / 2 for b in remaining_boxes]
                                gaps = [
                                    x_centers[k + 1] - x_centers[k]
                                    for k in range(len(x_centers) - 1)
                                ]
                                std_gap = float(np.std(gaps)) if len(gaps) > 1 else 0.0
                                if std_gap < best_std:
                                    best_std  = std_gap
                                    best_imgs = [
                                        img for j, img in enumerate(char_imgs)
                                        if j != drop_i
                                    ]
                        if best_imgs is not None:
                            char_imgs = best_imgs
                            n_seg    -= 1

            if n_seg != n_label:
                skipped_plates += 1
                continue

            for char_img, ch in zip(char_imgs, label_chars):
                idx = char_to_idx(ch)
                if idx is not None:
                    X_list.append(char_img)
                    y_list.append(idx)

    successful_plates = total_plates - skipped_plates
    log.info(
        "Collected %d character samples from %d plates "
        "(%d plates skipped due to segmentation mismatch).",
        len(X_list), successful_plates, skipped_plates,
    )

    if not X_list:
        raise RuntimeError(
            "No character samples could be extracted. "
            "Check the dataset path and image filenames."
        )

    X = np.array(X_list, dtype=np.uint8)   # (N, H, W)
    y = np.array(y_list, dtype=np.int64)
    seg_stats = {
        "total_plates":      total_plates,
        "successful_plates": successful_plates,
        "skipped_plates":    skipped_plates,
        "seg_success_rate":  successful_plates / max(total_plates, 1),
    }
    return X, y, seg_stats


# ---------------------------------------------------------------------------
# 3. PyTorch Dataset & DataLoaders
# ---------------------------------------------------------------------------

class PlateCharDataset(Dataset):
    """
    PyTorch Dataset wrapping the flat (X, y) character arrays.

    Each sample is a float32 tensor of shape (1, img_size, img_size)
    normalised to [0, 1].
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.X = X
        self.y = y
        self.augment = augment

        base_tf = [
            transforms.ToPILImage(),
            transforms.ToTensor(),               # → [0, 1] float32
            transforms.Normalize((0.5,), (0.5,)) # → [-1, 1]
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
            # Fix 3c – perspective distortion mimics off-angle camera shots
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            # Fix 3c – slightly higher erase prob for more robustness
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0),
        ]
        self.tf_base = transforms.Compose(base_tf)
        self.tf_aug  = transforms.Compose(aug_tf)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.X[idx]                        # (H, W) uint8
        tf  = self.tf_aug if self.augment else self.tf_base
        tensor = tf(img)                         # (1, H, W) float32
        return tensor, int(self.y[idx])


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = BATCH_SIZE,
    val_fraction: float = 0.15,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / validation / test DataLoaders.

    The training set is further split: (1 - val_fraction) for actual training,
    val_fraction for online validation during training.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    # Split train → train + val
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=val_fraction,
            random_state=seed,
            stratify=y_train,
        )
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=val_fraction,
            random_state=seed,
        )

    train_ds = PlateCharDataset(X_tr, y_tr, augment=True)
    val_ds   = PlateCharDataset(X_val, y_val, augment=False)
    test_ds  = PlateCharDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    log.info(
        "DataLoaders | train=%d  val=%d  test=%d  (batch=%d)",
        len(train_ds), len(val_ds), len(test_ds), batch_size,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 4. CNN Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU → MaxPool block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CharCNN(nn.Module):
    """
    CNN character classifier.

    Architecture (input: 1 × 28 × 28)
    -----------------------------------
    Conv Block 1 : 1   → 32  channels, MaxPool → 14×14
    Conv Block 2 : 32  → 64  channels, MaxPool →  7×7
    Conv Block 3 : 64  → 128 channels, no pool  →  7×7
    Conv Block 4 : 128 → 256 channels, no pool  →  7×7   (Fix 3b – deeper features)
    Flatten       : 256 × 7 × 7 = 12544
    FC1 (256) + Dropout
    FC2 (128) + Dropout
    FC3 (num_classes, softmax via CrossEntropyLoss)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        img_size: int = IMG_SIZE,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 128, pool=False),   # Fix 3b
        )

        # Dynamically compute the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            flat_size = self.features(dummy).numel()

        # Smaller FC head → less capacity → less overfitting on small datasets
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; returns raw logits (use with nn.CrossEntropyLoss)."""
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the training and validation loop for a given model.

    Attributes
    ----------
    train_losses : list[float]   – per-epoch average training loss
    val_losses   : list[float]   – per-epoch average validation loss
    val_accs     : list[float]   – per-epoch validation accuracy
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device
        self.scheduler = scheduler

        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []
        self.val_accs:     List[float] = []

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        """
        Run one epoch (train or eval).

        Returns
        -------
        avg_loss : float
        accuracy : float   (0–1)
        """
        self.model.train(train)
        total_loss = 0.0
        correct    = 0
        total      = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds       = logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += images.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = EPOCHS,
        early_stop_patience: int = EARLY_STOP_PATIENCE,
    ) -> None:
        """Train for *epochs* epochs with early stopping and record history."""
        log.info("Starting training on %s for up to %d epochs.", self.device, epochs)
        best_val_loss  = float("inf")
        best_state:    dict | None = None
        patience_count = 0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            vl_loss, vl_acc = self._run_epoch(val_loader,   train=False)

            self.train_losses.append(tr_loss)
            self.val_losses.append(vl_loss)
            self.val_accs.append(vl_acc)

            if self.scheduler is not None:
                self.scheduler.step(vl_loss)

            # Keep best model weights / early stopping counter
            if vl_loss < best_val_loss - 1e-4:
                best_val_loss  = vl_loss
                best_state     = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            log.info(
                "Epoch %3d/%d | "
                "train_loss=%.4f  train_acc=%.2f%%  |  "
                "val_loss=%.4f  val_acc=%.2f%%  "
                "[patience %d/%d]",
                epoch, epochs,
                tr_loss, tr_acc * 100,
                vl_loss, vl_acc * 100,
                patience_count, early_stop_patience,
            )

            if patience_count >= early_stop_patience:
                log.info(
                    "Early stopping triggered at epoch %d "
                    "(no val-loss improvement for %d epochs).",
                    epoch, early_stop_patience,
                )
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            log.info("Restored best model (val_loss=%.4f).", best_val_loss)


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluates a trained model on a DataLoader and reports
    accuracy, precision, recall, and F1-score.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model  = model
        self.device = device

    def evaluate(self, loader: DataLoader) -> dict:
        """
        Run inference and compute classification metrics.

        Returns
        -------
        metrics : dict with keys accuracy, precision, recall, f1
        """
        self.model.eval()
        all_preds:  List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                logits = self.model(images)
                preds  = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds,
                               average="weighted", zero_division=0)
        rec  = recall_score(all_labels, all_preds,
                            average="weighted", zero_division=0)
        f1   = f1_score(all_labels, all_preds,
                        average="weighted", zero_division=0)

        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        # Per-class F1 for confusable character pairs (stored in metrics dict
        # so the CSV export can include them without re-running inference).
        per_class_f1 = f1_score(
            all_labels, all_preds,
            labels=list(range(NUM_CLASSES)),
            average=None,
            zero_division=0,
        )
        for ch in ["B", "8", "O", "0", "S", "5", "I", "1", "Q", "Z", "2"]:
            idx = char_to_idx(ch)
            if idx is not None and idx < len(per_class_f1):
                metrics[f"f1_{ch}"] = float(per_class_f1[idx])

        # Pretty print
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)
        print(f"  Accuracy  : {acc  * 100:.2f}%")
        print(f"  Precision : {prec * 100:.2f}%  (weighted)")
        print(f"  Recall    : {rec  * 100:.2f}%  (weighted)")
        print(f"  F1-Score  : {f1   * 100:.2f}%  (weighted)")
        print("=" * 60)

        # Per-class report (only classes that appear in the test set)
        present_classes = sorted(set(all_labels) | set(all_preds))
        target_names    = [idx_to_char(i) for i in present_classes]
        print("\nPer-class report:")
        print(
            classification_report(
                all_labels, all_preds,
                labels=present_classes,
                target_names=target_names,
                zero_division=0,
            )
        )

        return metrics


# ---------------------------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path | None = None,
) -> None:
    """
    Plot training and validation loss curves side-by-side.

    Parameters
    ----------
    train_losses : per-epoch training loss
    val_losses   : per-epoch validation loss
    save_path    : if provided, save the figure; otherwise show interactively.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Training loss",   marker="o", markersize=3)
    ax.plot(epochs, val_losses,   label="Validation loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150)
        log.info("Loss curve saved → %s", save_path)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        log.info("GPU detected: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    log.info("No GPU detected, using CPU.")
    return torch.device("cpu")


def download_dataset() -> Path:
    """
    Download the Kaggle dataset via kagglehub and return the root path.

    The returned path points to the directory that contains the
    'dataset_final' folder (or is that folder itself).
    """
    import kagglehub  # imported here so the rest of the file works without it

    log.info("Downloading dataset from Kaggle …")
    raw_path = kagglehub.dataset_download(
        "abdelhamidzakaria/european-license-plates-dataset"
    )
    root = Path(raw_path)
    log.info("Dataset cached at: %s", root)
    return root


def find_dataset_root(base: Path) -> Path:
    """
    Walk *base* to locate the directory that contains train / val / test
    sub-folders.  Returns that directory.
    """
    # Direct match
    for candidate in [base, base / "dataset_final"]:
        if (candidate / "train").exists():
            return candidate

    # Recursive search
    for p in base.rglob("train"):
        if p.is_dir():
            return p.parent

    raise FileNotFoundError(
        f"Could not find 'train' subdirectory under {base}. "
        "Please verify the dataset structure."
    )


def main() -> None:
    """End-to-end pipeline: download → segment → train → evaluate → plot."""

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    base_path    = download_dataset()
    dataset_root = find_dataset_root(base_path)
    log.info("Using dataset root: %s", dataset_root)

    # ── 2. Character segmentation ───────────────────────────────────────────
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
    X, y, seg_stats = build_char_dataset(dataset_root, segmenter)

    # ── 3. Train / test split (80 : 20) ────────────────────────────────────
    # Remove classes that have fewer than 2 samples (cannot be stratified)
    from collections import Counter
    counts = Counter(y.tolist())
    keep_mask = np.array([counts[label] >= 2 for label in y.tolist()])
    removed = int((~keep_mask).sum())
    if removed:
        log.warning(
            "Dropped %d samples whose class has fewer than 2 members "
            "(cannot stratify). Classes dropped: %s",
            removed,
            sorted({idx_to_char(lbl) for lbl, cnt in counts.items() if cnt < 2}),
        )
    X, y = X[keep_mask], y[keep_mask]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=SEED, stratify=y,
        )
    except ValueError as exc:
        log.warning("Stratified split failed (%s); falling back to random split.", exc)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=SEED,
        )
    log.info(
        "Dataset split | train=%d  test=%d  classes=%d",
        len(y_train), len(y_test), NUM_CLASSES,
    )

    # ── 4. DataLoaders ──────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_test, y_test,
        batch_size=BATCH_SIZE,
    )

    # ── 5. Model ────────────────────────────────────────────────────────────
    device = get_device()
    model  = CharCNN(num_classes=NUM_CLASSES, img_size=IMG_SIZE, dropout=DROPOUT_RATE)
    model  = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d", total_params)

    # ── 6. Optimizer, loss, scheduler ───────────────────────────────────────
    # Fix 3a – class-frequency weighted loss to address rare-character under-training
    # (characters like 'I', 'O', 'Q', 'X' that appear rarely get higher weight)
    class_counts_map = Counter(y_train.tolist())
    weight_list = [
        float(len(y_train)) / (NUM_CLASSES * max(class_counts_map.get(c, 1), 1))
        for c in range(NUM_CLASSES)
    ]
    class_weights = torch.tensor(weight_list, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
        weight=class_weights,
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── 7. Training ─────────────────────────────────────────────────────────
    trainer = Trainer(model, optimizer, criterion, device, scheduler)
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    # ── 8. Save model weights ───────────────────────────────────────────────
    weights_path = Path("char_cnn_weights.pth")
    torch.save(model.state_dict(), weights_path)
    log.info("Model weights saved → %s", weights_path)

    # ── 9. Evaluation ───────────────────────────────────────────────────────
    evaluator    = Evaluator(model, device)
    char_metrics = evaluator.evaluate(test_loader)

    # ── 10. Loss curves ─────────────────────────────────────────────────────
    curve_path = Path("loss_curves_cnn.png")
    plot_loss_curves(
        trainer.train_losses,
        trainer.val_losses,
        save_path=curve_path,
    )

    # ── 11. Plate-level prediction demo (20 plates) ─────────────────────────
    predict_plates_demo(
        dataset_root=dataset_root,
        segmenter=segmenter,
        model=model,
        device=device,
        n_plates=20,
    )

    # ── 12. Full plate-level evaluation & CSV export ─────────────────────────
    plate_metrics = evaluate_plates(dataset_root, segmenter, model, device)

    all_metrics: dict = {
        # ── Identity ────────────────────────────────────────────────────────
        "model_type":               "CNN",
        "model_trainable_params":   total_params,
        # ── Character-level (test set) ───────────────────────────────────────
        "char_accuracy":            round(char_metrics["accuracy"],  4),
        "char_precision_weighted":  round(char_metrics["precision"], 4),
        "char_recall_weighted":     round(char_metrics["recall"],    4),
        "char_f1_weighted":         round(char_metrics["f1"],        4),
        # Per-class F1 for visually confusable pairs
        **{
            f"char_f1_{ch}": round(char_metrics.get(f"f1_{ch}", 0.0), 4)
            for ch in ["B", "8", "O", "0", "S", "5", "I", "1", "Q", "Z", "2"]
        },
        # ── Plate-level (all plates) ─────────────────────────────────────────
        "plate_exact_match_acc":    round(plate_metrics["exact_match_acc"],   4),
        "plate_mean_cer":           round(plate_metrics["mean_cer"],          4),
        "plate_mean_inference_ms":  round(plate_metrics["mean_inference_ms"], 2),
        "plate_n_evaluated":        plate_metrics["n_plates"],
        # ── Segmentation ────────────────────────────────────────────────────
        "seg_total_plates":         seg_stats["total_plates"],
        "seg_successful_plates":    seg_stats["successful_plates"],
        "seg_success_rate":         round(seg_stats["seg_success_rate"], 4),
    }

    save_metrics_csv(all_metrics, Path("cnn_metrics.csv"))
    log.info("Done.")


# ---------------------------------------------------------------------------
# 9. Plate-level prediction demo
# ---------------------------------------------------------------------------

def predict_plate(
    bgr: np.ndarray,
    segmenter: CharacterSegmenter,
    model: nn.Module,
    device: torch.device,
) -> str:
    """
    Predict the full plate text from a raw plate image.

    Parameters
    ----------
    bgr       : plate image (BGR numpy array)
    segmenter : CharacterSegmenter instance
    model     : trained CharCNN
    device    : torch device

    Returns
    -------
    predicted text string (e.g. 'AB1234')
    """
    char_imgs, _ = segmenter.segment(bgr, expected_count=None)

    # Fix 2b – if too many characters found, progressively tighten the
    # segmenter and re-run until we land in [MIN, MAX] or run out of attempts.
    if len(char_imgs) > MAX_PLATE_CHARS:
        tight_seg = CharacterSegmenter(
            img_size=segmenter.img_size,
            min_height_ratio=segmenter.min_height_ratio,
            min_area=segmenter.min_area,
            max_width_ratio=segmenter.max_width_ratio,
            nms_iou_threshold=segmenter.nms_iou_threshold,
            min_char_aspect=segmenter.min_char_aspect,
            max_char_aspect=segmenter.max_char_aspect,
            vert_center_tol=segmenter.vert_center_tol,
            border_exclusion_ratio=segmenter.border_exclusion_ratio,
            min_width_ratio_vs_median=segmenter.min_width_ratio_vs_median,
            min_height_ratio_vs_median=segmenter.min_height_ratio_vs_median,
            max_height_ratio_vs_median=segmenter.max_height_ratio_vs_median,
        )
        for _ in range(MAX_TIGHTEN_ATTEMPTS):
            tight_seg.max_width_ratio  = max(0.10, tight_seg.max_width_ratio  - 0.02)
            tight_seg.vert_center_tol  = max(0.05, tight_seg.vert_center_tol  - 0.03)
            new_imgs, _ = tight_seg.segment(bgr, expected_count=None)
            if len(new_imgs) <= MAX_PLATE_CHARS:
                char_imgs = new_imgs
                break

    if not char_imgs:
        return "<no_chars>"

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    model.eval()
    predicted_chars: List[str] = []
    with torch.no_grad():
        for patch in char_imgs:
            tensor = tf(patch).unsqueeze(0).to(device)   # (1,1,H,W)
            logit  = model(tensor)
            idx    = int(logit.argmax(dim=1).item())
            predicted_chars.append(idx_to_char(idx))

    return "".join(predicted_chars)


def predict_plates_demo(
    dataset_root: Path,
    segmenter: CharacterSegmenter,
    model: nn.Module,
    device: torch.device,
    n_plates: int = 20,
    img_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> None:
    """
    Run plate-level prediction on *n_plates* random plates and print
    a side-by-side comparison of the actual vs predicted plate text.

    Parameters
    ----------
    dataset_root : directory containing train / val / test sub-folders
    segmenter    : CharacterSegmenter instance
    model        : trained CharCNN
    device       : torch device
    n_plates     : number of plates to sample
    """
    # Collect all available plate images
    all_paths: List[Path] = []
    for sub in ("train", "val", "test"):
        folder = dataset_root / sub
        if folder.exists():
            all_paths.extend(
                p for p in folder.iterdir()
                if p.suffix.lower() in img_extensions
            )

    if not all_paths:
        log.warning("No plate images found for demo.")
        return

    rng = random.Random(SEED)
    sample = rng.sample(all_paths, min(n_plates, len(all_paths)))

    print("\n" + "=" * 62)
    print(f"  PLATE PREDICTION DEMO  ({len(sample)} plates)")
    print("=" * 62)
    print(f"  {'#':>3}  {'Actual':<12}  {'Predicted':<12}  {'Match?'}")
    print("  " + "-" * 58)

    correct = 0
    for i, img_path in enumerate(sample, start=1):
        actual = extract_label_from_filename(img_path)
        bgr    = cv2.imread(str(img_path))
        if bgr is None:
            predicted = "<read_error>"
        else:
            predicted = predict_plate(bgr, segmenter, model, device)

        match = "Y" if predicted == actual else "X"
        if predicted == actual:
            correct += 1
        print(f"  {i:>3}  {actual:<12}  {predicted:<12}  {match}")

    print("  " + "-" * 58)
    print(f"  Plate-level accuracy: {correct}/{len(sample)} "
          f"({correct / len(sample) * 100:.1f}%)")
    print("=" * 62)


# ---------------------------------------------------------------------------
# 10. Full plate-level evaluation (all plates) & CSV export
# ---------------------------------------------------------------------------

def evaluate_plates(
    dataset_root: Path,
    segmenter: CharacterSegmenter,
    model: nn.Module,
    device: torch.device,
    img_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> dict:
    """
    Run plate-level inference over every image in the dataset and compute:
      - Exact plate match accuracy
      - Mean Character Error Rate  (CER = edit_distance / label_length)
      - Mean inference time per plate (milliseconds)

    Returns a dict with keys: exact_match_acc, mean_cer,
    mean_inference_ms, n_plates.
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
        log.warning("evaluate_plates: no images found.")
        return {"exact_match_acc": 0.0, "mean_cer": 1.0,
                "mean_inference_ms": 0.0, "n_plates": 0}

    exact_matches = 0
    cer_values:   List[float] = []
    infer_times:  List[float] = []

    log.info("Evaluating plate-level metrics on %d plates …", len(all_paths))
    for img_path in all_paths:
        actual = extract_label_from_filename(img_path)
        bgr    = cv2.imread(str(img_path))
        if bgr is None:
            cer_values.append(1.0)
            continue

        t0        = time.perf_counter()
        predicted = predict_plate(bgr, segmenter, model, device)
        t1        = time.perf_counter()
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


def save_metrics_csv(metrics: dict, path: Path) -> None:
    """Write a single-row CSV with all metric keys as column headers."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    log.info("Metrics saved → %s", path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()