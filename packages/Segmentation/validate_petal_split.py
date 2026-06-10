"""
Petal Split Validation
=======================
Validate the Column centroid-based petal splitting method against
existing 8-class GT masks.

Process:
  1. Load 8-class GT mask
  2. Merge Petal_L (6) + Petal_R (7) → Petal (5) to simulate 6-class output
  3. Run petal_split() to reconstruct L/R
  4. Compare reconstructed vs original GT → IoU, accuracy, misclassification rate
  5. Save visualization

Usage:
    cd packages/Segmentation
    python validate_petal_split.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from petal_split import merge_petal_to_6class, split_petal

# ── Config ──
GT_MASK_DIR = "./Inference/test_datasets_masks"
ORIG_IMG_DIR = "./Inference/test_datasets"
OUTPUT_DIR = "./petal_split_validation"

CLASS_PETAL_L = 6
CLASS_PETAL_R = 7


def iou(mask_a, mask_b):
    inter = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    return inter / union if union > 0 else 1.0


def pixel_accuracy(pred, gt, region):
    """Accuracy within the petal region only."""
    if region.sum() == 0:
        return 1.0
    return np.sum(pred[region] == gt[region]) / region.sum()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mask_dir = Path(GT_MASK_DIR)
    mask_files = sorted(mask_dir.glob("*.png"))
    print(f"Found {len(mask_files)} GT masks in {GT_MASK_DIR}")

    results = []

    for mask_path in mask_files:
        stem = mask_path.stem

        # 1. Load 8-class GT
        gt_8class = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_8class is None:
            continue

        gt_petal_l = gt_8class == CLASS_PETAL_L
        gt_petal_r = gt_8class == CLASS_PETAL_R
        gt_petal_all = gt_petal_l | gt_petal_r

        # Skip if no petal in this mask
        if gt_petal_all.sum() == 0:
            print(f"  {stem}: No petal region, skipping")
            continue

        # 2. Simulate 6-class (merge L+R → class 5)
        mask_6class = merge_petal_to_6class(gt_8class)

        # 3. Run petal splitting
        reconstructed, split_cx, method = split_petal(mask_6class)

        recon_petal_l = reconstructed == CLASS_PETAL_L
        recon_petal_r = reconstructed == CLASS_PETAL_R

        # 4. Metrics
        iou_l = iou(recon_petal_l, gt_petal_l)
        iou_r = iou(recon_petal_r, gt_petal_r)
        iou_mean = (iou_l + iou_r) / 2.0

        acc = pixel_accuracy(reconstructed, gt_8class, gt_petal_all)

        # Misclassification: L pixels classified as R, and vice versa
        l_as_r = np.sum(gt_petal_l & recon_petal_r)
        r_as_l = np.sum(gt_petal_r & recon_petal_l)
        total_petal_px = gt_petal_all.sum()
        misclass_rate = (l_as_r + r_as_l) / total_petal_px

        results.append({
            "filename": mask_path.name,
            "method": method,
            "split_cx": split_cx,
            "iou_L": round(iou_l, 4),
            "iou_R": round(iou_r, 4),
            "iou_mean": round(iou_mean, 4),
            "accuracy": round(acc, 4),
            "misclass_rate": round(misclass_rate, 4),
            "petal_L_px": int(gt_petal_l.sum()),
            "petal_R_px": int(gt_petal_r.sum()),
            "L_as_R_px": int(l_as_r),
            "R_as_L_px": int(r_as_l),
        })

        # 5. Visualization (1×3 panel)
        h, w = gt_8class.shape

        # Panel 1: GT Petal_L (magenta) + Petal_R (orange)
        gt_vis = np.full((h, w, 3), 50, dtype=np.uint8)
        gt_vis[gt_petal_l] = [255, 0, 255]   # magenta
        gt_vis[gt_petal_r] = [255, 165, 0]    # orange

        # Panel 2: Reconstructed
        recon_vis = np.full((h, w, 3), 50, dtype=np.uint8)
        recon_vis[recon_petal_l] = [255, 0, 255]
        recon_vis[recon_petal_r] = [255, 165, 0]
        # Draw split line
        if split_cx is not None:
            cv2.line(recon_vis, (split_cx, 0), (split_cx, h), (0, 255, 0), 3)

        # Panel 3: Diff (white=correct, red=wrong)
        diff_vis = np.full((h, w, 3), 30, dtype=np.uint8)
        correct = gt_petal_all & (reconstructed == gt_8class)
        wrong = gt_petal_all & (reconstructed != gt_8class)
        diff_vis[correct] = [220, 220, 220]
        diff_vis[wrong] = [0, 0, 255]

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        axes[0].imshow(gt_vis[:, :, ::-1])
        axes[0].set_title("GT: Petal_L (magenta) + Petal_R (orange)", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(recon_vis[:, :, ::-1])
        axes[1].set_title(
            f"Reconstructed (method={method})\nsplit_cx={split_cx}",
            fontsize=12, fontweight='bold'
        )
        axes[1].axis('off')

        axes[2].imshow(diff_vis[:, :, ::-1])
        axes[2].set_title(
            f"Diff — IoU_L={iou_l:.3f} IoU_R={iou_r:.3f}\n"
            f"Accuracy={acc:.3f} MisclassRate={misclass_rate:.3f}",
            fontsize=11, fontweight='bold'
        )
        axes[2].axis('off')

        fig.suptitle(f"{stem}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{stem}_petal_split.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        status = "OK" if iou_mean > 0.85 else "!!"
        print(f"  {status} {stem}: IoU_L={iou_l:.3f} IoU_R={iou_r:.3f} mean={iou_mean:.3f} acc={acc:.3f} misclass={misclass_rate:.3f}")

    # Summary
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "petal_split_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(df)} masks)")
    print(f"{'='*60}")
    print(f"  Mean IoU_L:         {df['iou_L'].mean():.4f}")
    print(f"  Mean IoU_R:         {df['iou_R'].mean():.4f}")
    print(f"  Mean IoU (L+R):     {df['iou_mean'].mean():.4f}")
    print(f"  Mean Accuracy:      {df['accuracy'].mean():.4f}")
    print(f"  Mean MisclassRate:  {df['misclass_rate'].mean():.4f}")
    print(f"  Masks with IoU>0.85: {(df['iou_mean'] > 0.85).sum()}/{len(df)}")
    print(f"  Masks with IoU>0.90: {(df['iou_mean'] > 0.90).sum()}/{len(df)}")
    print(f"\n  CSV: {csv_path}")
    print(f"  Viz: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
