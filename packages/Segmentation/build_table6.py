"""
Build Table 6 (segmentation architecture comparison) from the four eval outputs.
======================================================================
Reads each model's summary.json + confusion_matrix.csv and prints:
  1. A markdown table (Global Pixel Acc / mAcc / mIoU / per-class IoU) ready to paste.
  2. The Petal_L<->Petal_R confusion (recall-normalized off-diagonal) — the direct
     evidence for the CoordConv-X claim.

Run AFTER the 4 evaluations have produced their summary.json + confusion_matrix.csv:
    uv run build_table6.py
"""

import os
import csv
import json

# (label, eval output dir) — order = column order in the table.
# CoordConv-X was dropped from the paper (did not beat plain DINOv3); left on disk for reference.
MODELS = [
    ("DINOv3 ViT-L (proposed)", "./Evaluation/Finalexp_v1_eval-dataset"),
    ("DeepLabV3 ResNet-50",     "./Evaluation/baseline_deeplabv3_eval-dataset"),
    ("U-Net (ResNet-34, ImageNet)", "./Evaluation/baseline_unet_eval-dataset"),
    # ("CoordConv-X (dropped)", "./Evaluation/CoordConvX_eval-dataset"),
]

# Rows to show (skip the vestigial "Petal" class which is always absent)
CLASS_ROWS = ["Background", "Column", "Dorsal Sepal", "Labellum",
              "Lateral Sepal", "Petal_L", "Petal_R"]
CLASS_NAMES = ["Background", "Column", "Dorsal Sepal", "Labellum",
               "Lateral Sepal", "Petal", "Petal_L", "Petal_R"]


def load_summary(d):
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def load_cm(d):
    """Return dict[(true,pred)] = count, or None."""
    p = os.path.join(d, "confusion_matrix.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.reader(open(p)))
    header = rows[0][1:]
    cm = {}
    for r in rows[1:]:
        true = r[0]
        for j, v in enumerate(r[1:]):
            cm[(true, header[j])] = float(v)
    return cm


def lr_confusion(cm):
    """Recall-normalized L->R and R->L (% of that true class's pixels)."""
    if cm is None:
        return None, None
    l_total = sum(cm.get(("Petal_L", c), 0) for c in CLASS_NAMES)
    r_total = sum(cm.get(("Petal_R", c), 0) for c in CLASS_NAMES)
    l2r = 100 * cm.get(("Petal_L", "Petal_R"), 0) / l_total if l_total else float("nan")
    r2l = 100 * cm.get(("Petal_R", "Petal_L"), 0) / r_total if r_total else float("nan")
    return l2r, r2l


def main():
    summaries, cms, labels = [], [], []
    for label, d in MODELS:
        s = load_summary(d)
        if s is None:
            print(f"⚠️  missing: {d}/summary.json — skipping ({label})")
            continue
        summaries.append(s); cms.append(load_cm(d)); labels.append(label)

    if not summaries:
        print("No eval outputs found. Run the evaluations first.")
        return

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)

    # ---- main metrics table ----
    print("\n### Table 6 — segmentation comparison (eval-dataset, 9 held-out images)\n")
    cols = "| Metric | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    print(cols); print(sep)
    for metric in ["Global Pixel Acc", "mAcc", "mIoU"]:
        print(f"| {metric} | " + " | ".join(fmt(s[metric]) for s in summaries) + " |")
    for cls in CLASS_ROWS:
        print(f"| IoU — {cls} | " + " | ".join(fmt(s["IoU_per_class"][cls]) for s in summaries) + " |")

    # ---- L/R confusion ----
    print("\n### Petal_L <-> Petal_R confusion (recall-normalized, % of that true class)\n")
    print("| Direction | " + " | ".join(labels) + " |")
    print(sep)
    l2r_vals, r2l_vals = [], []
    for cm in cms:
        l2r, r2l = lr_confusion(cm)
        l2r_vals.append(l2r); r2l_vals.append(r2l)
    def pct(v):
        return f"{v:.2f}%" if (v is not None and v == v) else "n/a"
    print("| true Petal_L → pred Petal_R | " + " | ".join(pct(v) for v in l2r_vals) + " |")
    print("| true Petal_R → pred Petal_L | " + " | ".join(pct(v) for v in r2l_vals) + " |")
    print("\n(If confusion_matrix.csv is missing for a model, re-run its evaluation script — "
          "the CSV export was added on 2026-06-28.)")


if __name__ == "__main__":
    main()
