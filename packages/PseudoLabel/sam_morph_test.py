"""
SAM2 Morphological Kernel Size Experiment
==========================================
YOLO + SAM2 → 比較不同 Morph Close kernel size 對 mask 品質的影響。

Kernel sizes: 30×30, 70×70, 120×120, 200×200

每張圖產出一張 1×6 大圖:
  Col 1: Original + YOLO box
  Col 2: Raw SAM2 mask (no post-processing)
  Col 3-6: Morph Close at each kernel size

Usage:
    cd packages/PseudoLabel
    python sam_morph_test.py
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
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
INPUT_DIR = "./inputs"
OUTPUT_DIR = "./outputs/sam_morph_test"

YOLO_WEIGHTS = "../Detection/training_results/flower_detect/FD11/weights/best.pt"
YOLO_IMG_SIZE = 960
YOLO_CONF = 0.5
YOLO_IOU = 0.75

SAM2_MODEL = "facebook/sam2-hiera-large"

SPTS_ALPHA = 1.0
SPTS_BETA = 1.0

KERNEL_SIZES = [120, 200, 250, 300, 350]


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


def morph_close(mask_bool, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m = mask_bool.astype(np.uint8) * 255
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return closed > 0


def overlay_mask(image_bgr, mask, color=(0, 255, 0), alpha=0.4):
    vis = image_bgr.copy()
    overlay = np.zeros_like(vis)
    overlay[mask] = color
    return cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)


def pixel_diff(mask_a, mask_b):
    """Count pixels that differ."""
    return int(np.sum(mask_a != mask_b))


def iou_score(m1, m2):
    inter = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    return inter / union if union > 0 else 0.0


# ==========================================
# Main
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_dir = Path(INPUT_DIR)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    img_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    print(f"Found {len(img_paths)} images in {INPUT_DIR}")

    if not img_paths:
        print("No images found. Put test photos in ./inputs/")
        return

    # Load models
    print(f"Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)

    print(f"Loading SAM2 ({SAM2_MODEL})...")
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam = SAM2ImagePredictor.from_pretrained(SAM2_MODEL)
    if hasattr(sam, "to"):
        sam.to(DEVICE)
    print("Models loaded.\n")

    n_cols = 2 + len(KERNEL_SIZES)  # original + raw + N kernels

    # Dynamic colors: green for raw, colormap for kernel variants
    raw_color = (0, 255, 0)
    cmap = plt.cm.get_cmap("tab10", len(KERNEL_SIZES))
    kernel_colors = [
        tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(len(KERNEL_SIZES))
    ]

    for img_path in img_paths:
        stem = img_path.stem
        print(f"Processing: {stem}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        # YOLO + SPTS
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

        # Draw YOLO
        img_yolo = img_bgr.copy()
        for box in boxes_all:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (200, 200, 200), 2)
        px1, py1, px2, py2 = primary_box[0].astype(int)
        cv2.rectangle(img_yolo, (px1, py1), (px2, py2), (0, 255, 0), 4)

        # SAM2 raw prediction
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        sam.set_image(image_rgb)
        box = primary_box[0].astype(np.float32)
        masks, scores, _ = sam.predict(
            point_coords=None, point_labels=None,
            box=box, multimask_output=False
        )
        mask_raw = masks[0]
        if mask_raw.dtype != np.bool_:
            mask_raw = mask_raw.astype(bool)
        raw_score = float(scores[0])
        raw_area = int(np.sum(mask_raw))

        # Morph close variants
        mask_variants = []
        for ks in KERNEL_SIZES:
            m = morph_close(mask_raw, ks)
            mask_variants.append(m)

        # ── Build figure: 1 × (2 + N_kernels) ──
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))

        # Col 0: Original + YOLO
        axes[0].imshow(img_yolo[:, :, ::-1])
        axes[0].set_title("YOLO + SPTS", fontsize=13, fontweight='bold')
        axes[0].axis('off')

        # Col 1: Raw SAM2
        axes[1].imshow(overlay_mask(img_bgr, mask_raw, raw_color)[:, :, ::-1])
        axes[1].set_title(f"Raw SAM2\nscore={raw_score:.3f}\narea={raw_area:,}px",
                          fontsize=12)
        axes[1].axis('off')

        # Col 2+: Morph Close variants
        for i, (ks, mv) in enumerate(zip(KERNEL_SIZES, mask_variants)):
            col = 2 + i
            filled_area = int(np.sum(mv))
            delta = filled_area - raw_area
            iou = iou_score(mask_raw, mv)

            axes[col].imshow(overlay_mask(img_bgr, mv, kernel_colors[i])[:, :, ::-1])
            axes[col].set_title(
                f"Close {ks}×{ks}\n"
                f"area={filled_area:,}px (Δ+{delta:,})\n"
                f"IoU vs raw={iou:.4f}",
                fontsize=12
            )
            axes[col].axis('off')

        fig.suptitle(
            f"Morph Close Kernel Size Comparison: {stem}\n"
            f"Image: {w}×{h}px",
            fontsize=16, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f"{stem}_morph_compare.png")
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

        # Print summary
        print(f"  Raw area: {raw_area:,}px")
        for ks, mv in zip(KERNEL_SIZES, mask_variants):
            fa = int(np.sum(mv))
            print(f"  Close {ks:3d}×{ks:3d}: area={fa:,}px  Δ=+{fa-raw_area:,}  IoU={iou_score(mask_raw, mv):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
