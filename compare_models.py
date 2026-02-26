"""
Comparative Analysis: CNN vs CRNN for License Plate Recognition
================================================================
Reads  cnn_metrics.csv  and  crnn_metrics.csv  (produced by the
respective training scripts) and prints a side-by-side comparison
of all shared metrics.  Also generates a bar-chart visualisation
saved as  comparison_chart.png.

Usage:
    python compare_models.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1.  Load metrics
# ---------------------------------------------------------------------------

def load_metrics(path: Path) -> dict:
    """Read single-row CSV into a dict, converting numeric strings."""
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run the corresponding "
                 f"training script first.")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    parsed: dict = {}
    for k, v in row.items():
        try:
            parsed[k] = int(v)
        except ValueError:
            try:
                parsed[k] = float(v)
            except ValueError:
                parsed[k] = v
    return parsed


# ---------------------------------------------------------------------------
# 2.  Print comparison table
# ---------------------------------------------------------------------------

SECTION_LABELS = {
    "model_type":               ("Model",            "Identity"),
    "model_trainable_params":   ("Trainable params", "Identity"),
    "char_accuracy":            ("Char accuracy",              "Character-level (test set)"),
    "char_precision_weighted":  ("Char precision (weighted)",  "Character-level (test set)"),
    "char_recall_weighted":     ("Char recall (weighted)",     "Character-level (test set)"),
    "char_f1_weighted":         ("Char F1 (weighted)",         "Character-level (test set)"),
    "char_f1_B":  ("F1  B", "Per-class F1 (confusable pairs)"),
    "char_f1_8":  ("F1  8", "Per-class F1 (confusable pairs)"),
    "char_f1_O":  ("F1  O", "Per-class F1 (confusable pairs)"),
    "char_f1_0":  ("F1  0", "Per-class F1 (confusable pairs)"),
    "char_f1_S":  ("F1  S", "Per-class F1 (confusable pairs)"),
    "char_f1_5":  ("F1  5", "Per-class F1 (confusable pairs)"),
    "char_f1_I":  ("F1  I", "Per-class F1 (confusable pairs)"),
    "char_f1_1":  ("F1  1", "Per-class F1 (confusable pairs)"),
    "char_f1_Q":  ("F1  Q", "Per-class F1 (confusable pairs)"),
    "char_f1_Z":  ("F1  Z", "Per-class F1 (confusable pairs)"),
    "char_f1_2":  ("F1  2", "Per-class F1 (confusable pairs)"),
    "plate_exact_match_acc":    ("Plate exact match",     "Plate-level (all plates)"),
    "plate_mean_cer":           ("Plate mean CER",        "Plate-level (all plates)"),
    "plate_mean_inference_ms":  ("Plate mean infer (ms)", "Plate-level (all plates)"),
    "plate_n_evaluated":        ("Plates evaluated",      "Plate-level (all plates)"),
    "seg_total_plates":         ("Total plates",        "Segmentation"),
    "seg_successful_plates":    ("Successful plates",   "Segmentation"),
    "seg_success_rate":         ("Seg success rate",    "Segmentation"),
}

# Metrics where LOWER is better
LOWER_IS_BETTER = {"plate_mean_cer", "plate_mean_inference_ms"}


def fmt(value) -> str:
    """Format a metric value for display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_comparison(cnn: dict, crnn: dict) -> None:
    """Print a nicely formatted comparison table."""
    col_w = 30
    val_w = 14

    header = (
        f"  {'Metric':<{col_w}}  {'CNN':>{val_w}}  {'CRNN':>{val_w}}  "
        f"{'Better':>8}"
    )
    sep = "  " + "-" * (col_w + 2 * val_w + 14)

    print("\n" + "=" * len(header))
    print("   COMPARATIVE ANALYSIS:  CNN  vs  CRNN")
    print("=" * len(header))

    prev_section = None
    for key in SECTION_LABELS:
        nice_name, section = SECTION_LABELS[key]

        if section != prev_section:
            print(sep)
            print(f"\n  [{section}]")
            prev_section = section

        c_val  = cnn.get(key, "n/a")
        cr_val = crnn.get(key, "n/a")

        # determine winner
        winner = ""
        if isinstance(c_val, (int, float)) and isinstance(cr_val, (int, float)):
            if key in LOWER_IS_BETTER:
                if cr_val < c_val:
                    winner = "CRNN"
                elif c_val < cr_val:
                    winner = "CNN"
                else:
                    winner = "tie"
            elif key not in ("plate_n_evaluated", "seg_total_plates",
                             "seg_successful_plates", "model_trainable_params"):
                if cr_val > c_val:
                    winner = "CRNN"
                elif c_val > cr_val:
                    winner = "CNN"
                else:
                    winner = "tie"

        print(
            f"  {nice_name:<{col_w}}  {fmt(c_val):>{val_w}}  "
            f"{fmt(cr_val):>{val_w}}  {winner:>8}"
        )

    print(sep)
    print()


# ---------------------------------------------------------------------------
# 3.  Bar chart
# ---------------------------------------------------------------------------

def plot_comparison(cnn: dict, crnn: dict, save_path: Path) -> None:
    """Generate a grouped bar chart comparing key numeric metrics."""

    # Select metrics to visualise
    chart_metrics = [
        ("char_accuracy",           "Char\nAccuracy"),
        ("char_precision_weighted", "Char\nPrecision"),
        ("char_recall_weighted",    "Char\nRecall"),
        ("char_f1_weighted",        "Char\nF1"),
        ("plate_exact_match_acc",   "Plate\nExact Match"),
        ("plate_mean_cer",          "Plate\nMean CER"),
    ]

    keys   = [k for k, _ in chart_metrics]
    labels = [l for _, l in chart_metrics]
    cnn_vals  = [float(cnn.get(k, 0))  for k in keys]
    crnn_vals = [float(crnn.get(k, 0)) for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, cnn_vals,  width, label="CNN",  color="#4C72B0")
    bars2 = ax.bar(x + width / 2, crnn_vals, width, label="CRNN", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("CNN vs CRNN — Key Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Value annotations
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"Comparison chart saved -> {save_path}")

    # ── Per-class F1 chart ─────────────────────────────────────────────────
    confusable = ["B", "8", "O", "0", "S", "5", "I", "1", "Q", "Z", "2"]
    c_f1  = [float(cnn.get(f"char_f1_{ch}", 0))  for ch in confusable]
    cr_f1 = [float(crnn.get(f"char_f1_{ch}", 0)) for ch in confusable]

    x2 = np.arange(len(confusable))
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.bar(x2 - width / 2, c_f1,  width, label="CNN",  color="#4C72B0")
    ax2.bar(x2 + width / 2, cr_f1, width, label="CRNN", color="#DD8452")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Per-class F1 — Confusable Character Pairs")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(confusable)
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()

    f1_path = save_path.parent / "comparison_per_class_f1.png"
    fig2.savefig(str(f1_path), dpi=150)
    plt.close(fig2)
    print(f"Per-class F1 chart saved -> {f1_path}")


# ---------------------------------------------------------------------------
# 4.  Summary analysis
# ---------------------------------------------------------------------------

def print_summary(cnn: dict, crnn: dict) -> None:
    """Print a brief textual summary of the comparison."""
    print("=" * 64)
    print("   SUMMARY")
    print("=" * 64)

    # Character-level winner
    cnn_char_f1  = float(cnn.get("char_f1_weighted", 0))
    crnn_char_f1 = float(crnn.get("char_f1_weighted", 0))
    if crnn_char_f1 > cnn_char_f1:
        print(f"  Character F1:  CRNN wins ({crnn_char_f1:.4f} vs {cnn_char_f1:.4f})")
    elif cnn_char_f1 > crnn_char_f1:
        print(f"  Character F1:  CNN wins ({cnn_char_f1:.4f} vs {crnn_char_f1:.4f})")
    else:
        print(f"  Character F1:  Tie ({cnn_char_f1:.4f})")

    # Plate-level winner
    cnn_plate  = float(cnn.get("plate_exact_match_acc", 0))
    crnn_plate = float(crnn.get("plate_exact_match_acc", 0))
    if crnn_plate > cnn_plate:
        print(f"  Plate Match:   CRNN wins ({crnn_plate:.4f} vs {cnn_plate:.4f})")
    elif cnn_plate > crnn_plate:
        print(f"  Plate Match:   CNN wins ({cnn_plate:.4f} vs {crnn_plate:.4f})")
    else:
        print(f"  Plate Match:   Tie ({cnn_plate:.4f})")

    # CER winner
    cnn_cer  = float(cnn.get("plate_mean_cer", 1))
    crnn_cer = float(crnn.get("plate_mean_cer", 1))
    if crnn_cer < cnn_cer:
        print(f"  Mean CER:      CRNN wins ({crnn_cer:.4f} vs {cnn_cer:.4f})")
    elif cnn_cer < crnn_cer:
        print(f"  Mean CER:      CNN wins ({cnn_cer:.4f} vs {crnn_cer:.4f})")
    else:
        print(f"  Mean CER:      Tie ({cnn_cer:.4f})")

    # Speed
    cnn_ms  = float(cnn.get("plate_mean_inference_ms", 0))
    crnn_ms = float(crnn.get("plate_mean_inference_ms", 0))
    if crnn_ms < cnn_ms:
        print(f"  Inference:     CRNN faster ({crnn_ms:.2f} ms vs {cnn_ms:.2f} ms)")
    elif cnn_ms < crnn_ms:
        print(f"  Inference:     CNN faster ({cnn_ms:.2f} ms vs {crnn_ms:.2f} ms)")
    else:
        print(f"  Inference:     Tie ({cnn_ms:.2f} ms)")

    # Model size
    cnn_p  = int(cnn.get("model_trainable_params", 0))
    crnn_p = int(crnn.get("model_trainable_params", 0))
    print(f"  Parameters:    CNN={cnn_p:,}  CRNN={crnn_p:,}")

    # Key observations about confusable pairs
    print("\n  Confusable-pair highlights:")
    for a, b in [("B", "8"), ("O", "0"), ("S", "5"), ("I", "1"), ("Z", "2")]:
        cnn_a  = float(cnn.get(f"char_f1_{a}", 0))
        crnn_a = float(crnn.get(f"char_f1_{a}", 0))
        cnn_b  = float(cnn.get(f"char_f1_{b}", 0))
        crnn_b = float(crnn.get(f"char_f1_{b}", 0))
        sum_cnn  = cnn_a + cnn_b
        sum_crnn = crnn_a + crnn_b
        better = "CRNN" if sum_crnn > sum_cnn else "CNN" if sum_cnn > sum_crnn else "tie"
        print(
            f"    {a}/{b}:  CNN=({cnn_a:.3f}, {cnn_b:.3f})  "
            f"CRNN=({crnn_a:.3f}, {crnn_b:.3f})  -> {better}"
        )

    # Overall verdict
    print("\n  -- Methodology note --")
    print("  CNN evaluates characters after segmentation (segmentation-based);")
    print("  CRNN evaluates characters from CTC-decoded plate strips (end-to-end).")
    print("  The CRNN bypasses segmentation errors but must solve alignment implicitly.")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cnn_path  = Path("cnn_metrics.csv")
    crnn_path = Path("crnn_metrics.csv")

    cnn_metrics  = load_metrics(cnn_path)
    crnn_metrics = load_metrics(crnn_path)

    print_comparison(cnn_metrics, crnn_metrics)
    plot_comparison(cnn_metrics, crnn_metrics, Path("comparison_chart.png"))
    print_summary(cnn_metrics, crnn_metrics)


if __name__ == "__main__":
    main()
