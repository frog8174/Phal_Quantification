"""
SAM2 Ablation Experiment
========================
獨立測試 YOLO + SAM 的 mask 品質，比較三個變因：

  Variable A: SAM Version     — SAM 2.0 vs SAM 2.1
  Variable B: multimask_output — False (single) vs True (best-of-3)
  Variable C: Post-processing  — None vs Morphological Close (fill holes)

每張輸入圖產出一張 3×4 大圖 (3 rows = 3 variables, 4 cols):
  Col 1: Original + YOLO box
  Col 2: Variable off
  Col 3: Variable on
  Col 4: Diff heatmap (off vs on)

Usage:
    cd packages/PseudoLabel
    python sam_ablation.py
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# Config
# ==========================================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
INPUT_DIR = "./inputs"
OUTPUT_DIR = "./outputs/sam_ablation"

# YOLO
YOLO_WEIGHTS = "../Detection/training_results/flower_detect/FD11/weights/best.pt"
YOLO_IMG_SIZE = 960
YOLO_CONF = 0.5
YOLO_IOU = 0.75

# SAM models
SAM2_V20 = "facebook/sam2-hiera-large"
SAM2_V21 = "facebook/sam2.1-hiera-large"

# Morphological kernel for post-processing experiment
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# SPTS
SPTS_ALPHA = 1.0
SPTS_BETA = 1.0


# ==========================================
# Helpers
# ==========================================
def select_primary_flower(boxes_xyxy, img_h, img_w):
    if len(boxes_xyxy) == 0:
        return None
    img_cx, img_cy = img_w / 2.0, img_h / 2.0
    img_diag = np.sqrt(img_w ** 2 + img_h ** 2)
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    max_area = areas.max()
    box_cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    box_cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    dists = np.sqrt((box_cx - img_cx) ** 2 + (box_cy - img_cy) ** 2)
    scores = SPTS_ALPHA * (areas / max_area) - SPTS_BETA * (dists / img_diag)
    best = np.argmax(scores)
    return boxes_xyxy[best : best + 1]


def load_sam_predictor(model_id, device):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    if hasattr(predictor, "to"):
        predictor.to(device)
    return predictor


def sam_predict(predictor, image_bgr, box_xyxy, multimask=False):
    """Run SAM prediction. Returns (mask_bool, score)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    box = box_xyxy[0].astype(np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=box, multimask_output=multimask
    )

    if multimask:
        best = np.argmax(scores)
        mask = masks[best]
        score = scores[best]
    else:
        mask = masks[0]
        score = scores[0]

    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    return mask, float(score)


def morph_close(mask_bool):
    """Apply morphological closing to fill holes."""
    m = mask_bool.astype(np.uint8) * 255
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, MORPH_KERNEL)
    return closed > 0


def overlay_mask(image_bgr, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay a colored mask on image."""
    vis = image_bgr.copy()
    overlay = np.zeros_like(vis)
    overlay[mask] = color
    return cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)


def diff_image(mask_a, mask_b):
    """Create a diff visualization: green=added, red=removed, gray=same."""
    h, w = mask_a.shape
    diff = np.full((h, w, 3), 128, dtype=np.uint8)  # gray background
    both = mask_a & mask_b
    only_a = mask_a & ~mask_b  # in A but not B (removed)
    only_b = mask_b & ~mask_a  # in B but not A (added)
    diff[both] = [200, 200, 200]      # white-ish = same
    diff[only_a] = [0, 0, 255]        # red = only in baseline
    diff[only_b] = [0, 255, 0]        # green = only in variant
    diff[~mask_a & ~mask_b] = [50, 50, 50]  # dark = neither
    return diff


def iou_score(m1, m2):
    inter = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    return inter / union if union > 0 else 0.0


# ==========================================
# Main
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect images
    input_dir = Path(INPUT_DIR)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    img_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    print(f"Found {len(img_paths)} images in {INPUT_DIR}")

    if not img_paths:
        print("No images found. Put test photos in ./inputs/")
        return

    # Load YOLO
    print(f"Loading YOLO from {YOLO_WEIGHTS}...")
    yolo = YOLO(YOLO_WEIGHTS)

    # Load SAM predictors
    print("Loading SAM 2.0...")
    sam_v20 = load_sam_predictor(SAM2_V20, DEVICE)
    print("Loading SAM 2.1...")
    sam_v21 = load_sam_predictor(SAM2_V21, DEVICE)
    print("All models loaded.\n")

    for img_path in img_paths:
        stem = img_path.stem
        print(f"Processing: {stem}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        # YOLO detection + SPTS
        results = yolo.predict(
            source=str(img_path), conf=YOLO_CONF, iou=YOLO_IOU,
            imgsz=YOLO_IMG_SIZE, max_det=100, device=DEVICE, verbose=False
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            print(f"  No detection, skipping.")
            continue

        boxes_all = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        primary_box = select_primary_flower(boxes_all, h, w)
        if primary_box is None:
            continue

        # Draw YOLO result
        img_yolo = img_bgr.copy()
        for box in boxes_all:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (200, 200, 200), 2)
        px1, py1, px2, py2 = primary_box[0].astype(int)
        cv2.rectangle(img_yolo, (px1, py1), (px2, py2), (0, 255, 0), 4)

        # ── Run all 6 experiments ──
        # Exp A: SAM version (v2.0 vs v2.1), multimask=False, no postproc
        mask_v20, score_v20 = sam_predict(sam_v20, img_bgr, primary_box, multimask=False)
        mask_v21, score_v21 = sam_predict(sam_v21, img_bgr, primary_box, multimask=False)

        # Exp B: multimask_output (False vs True), using SAM 2.0
        mask_single, score_single = mask_v20, score_v20  # already computed
        mask_multi, score_multi = sam_predict(sam_v20, img_bgr, primary_box, multimask=True)

        # Exp C: Post-processing (raw vs morph_close), using SAM 2.0
        mask_raw = mask_v20
        mask_filled = morph_close(mask_v20)

        # ── Build 3×4 figure ──
        fig, axes = plt.subplots(3, 4, figsize=(28, 21))

        row_labels = [
            "A: SAM Version",
            "B: multimask_output",
            "C: Post-processing",
        ]

        # --- Row 0: SAM 2.0 vs 2.1 ---
        axes[0][0].imshow(img_yolo[:, :, ::-1])
        axes[0][0].set_title("YOLO + SPTS", fontsize=12, fontweight='bold')

        axes[0][1].imshow(overlay_mask(img_bgr, mask_v20)[:, :, ::-1])
        axes[0][1].set_title(f"SAM 2.0 (score={score_v20:.3f})", fontsize=12)

        axes[0][2].imshow(overlay_mask(img_bgr, mask_v21, color=(255, 165, 0))[:, :, ::-1])
        axes[0][2].set_title(f"SAM 2.1 (score={score_v21:.3f})", fontsize=12)

        iou_a = iou_score(mask_v20, mask_v21)
        axes[0][3].imshow(diff_image(mask_v20, mask_v21)[:, :, ::-1])
        axes[0][3].set_title(f"Diff (IoU={iou_a:.3f})\nRed=2.0 only, Green=2.1 only", fontsize=10)

        # --- Row 1: single vs multimask ---
        axes[1][0].imshow(img_yolo[:, :, ::-1])
        axes[1][0].set_title("YOLO + SPTS", fontsize=12, fontweight='bold')

        axes[1][1].imshow(overlay_mask(img_bgr, mask_single)[:, :, ::-1])
        axes[1][1].set_title(f"single mask (score={score_single:.3f})", fontsize=12)

        axes[1][2].imshow(overlay_mask(img_bgr, mask_multi, color=(0, 165, 255))[:, :, ::-1])
        axes[1][2].set_title(f"best-of-3 mask (score={score_multi:.3f})", fontsize=12)

        iou_b = iou_score(mask_single, mask_multi)
        axes[1][3].imshow(diff_image(mask_single, mask_multi)[:, :, ::-1])
        axes[1][3].set_title(f"Diff (IoU={iou_b:.3f})\nRed=single only, Green=multi only", fontsize=10)

        # --- Row 2: raw vs morph close ---
        axes[2][0].imshow(img_yolo[:, :, ::-1])
        axes[2][0].set_title("YOLO + SPTS", fontsize=12, fontweight='bold')

        axes[2][1].imshow(overlay_mask(img_bgr, mask_raw)[:, :, ::-1])
        axes[2][1].set_title("Raw mask", fontsize=12)

        axes[2][2].imshow(overlay_mask(img_bgr, mask_filled, color=(255, 0, 255))[:, :, ::-1])
        axes[2][2].set_title("+ Morph Close (15×15)", fontsize=12)

        iou_c = iou_score(mask_raw, mask_filled)
        axes[2][3].imshow(diff_image(mask_raw, mask_filled)[:, :, ::-1])
        axes[2][3].set_title(f"Diff (IoU={iou_c:.3f})\nRed=raw only, Green=filled only", fontsize=10)

        # Row labels
        for r in range(3):
            axes[r][0].set_ylabel(row_labels[r], fontsize=14, fontweight='bold', rotation=0,
                                  labelpad=120, va='center')
            for c in range(4):
                axes[r][c].axis('off')

        fig.suptitle(f"SAM Ablation: {stem}", fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0.08, 0, 1, 0.96])

        save_path = os.path.join(OUTPUT_DIR, f"{stem}_sam_ablation.png")
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")
        print(f"  IoU — A(version): {iou_a:.3f}  B(multimask): {iou_b:.3f}  C(postproc): {iou_c:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
