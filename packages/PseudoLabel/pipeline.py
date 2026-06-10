"""
End-to-end Pseudo-Labeling Pipeline
====================================
YOLO → SPTS → SAM2 → DINOv3 + Linear Head → 8-class Pseudo Mask

Usage:
    cd packages/PseudoLabel
    uv run pipeline.py
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torchvision.transforms import v2
from transformers import AutoModel
from PIL import Image
from tqdm import tqdm

# pyrefly: ignore [missing-import]
from config import (
    DEVICE, INPUT_DIR, OUTPUT_DIR, CROP_DIR, MASK_DIR, VIZ_DIR, STEPS_DIR, LOG_DIR,
    YOLO_WEIGHTS, YOLO_IMG_SIZE, YOLO_CONF, YOLO_IOU, YOLO_MAX_DET,
    SPTS_ALPHA, SPTS_BETA,
    SAM2_MODEL,
    DINO_MODEL_NAME, DINO_CHECKPOINT, DINO_IMG_SIZE, DINO_PATCH_SIZE,
    DINO_NUM_CLASSES, DINO_EXTRACT_LAYERS,
    CLASS_NAMES, CLASS_COLORS,
)

# ==========================================
# 0. Directory setup
# ==========================================
for d in [CROP_DIR, MASK_DIR, VIZ_DIR, STEPS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


# ==========================================
# 1. YOLO + SPTS
# ==========================================
def select_primary_flower(boxes_xyxy, img_h, img_w):
    """
    Spatial Prominence-based Target Selection (SPTS).
    Score = alpha * norm_area - beta * norm_distance
    Returns the single box with the highest score.
    """
    if len(boxes_xyxy) == 0:
        return None

    img_cx, img_cy = img_w / 2.0, img_h / 2.0
    img_diag = np.sqrt(img_w ** 2 + img_h ** 2)

    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    max_area = areas.max()

    box_cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    box_cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    dists = np.sqrt((box_cx - img_cx) ** 2 + (box_cy - img_cy) ** 2)

    norm_area = areas / max_area
    norm_dist = dists / img_diag

    scores = SPTS_ALPHA * norm_area - SPTS_BETA * norm_dist
    best_idx = np.argmax(scores)
    return boxes_xyxy[best_idx : best_idx + 1]


def run_yolo_detection(model, img_path, device):
    """Run YOLO detection and return primary flower bbox."""
    results = model.predict(
        source=str(img_path),
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMG_SIZE,
        max_det=YOLO_MAX_DET,
        device=device,
        verbose=False
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, 0

    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
    n_detections = len(boxes_xyxy)
    primary_box = select_primary_flower(boxes_xyxy, *cv2.imread(str(img_path)).shape[:2])
    return primary_box, n_detections


# ==========================================
# 2. SAM2 Instance Segmentation
# ==========================================
class SAM2Wrapper:
    def __init__(self, device):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL)
        if hasattr(self.predictor, "to"):
            self.predictor.to(device)
        self.device = device

    def segment(self, image_bgr, box_xyxy, morph_kernel_size=200):
        """Given an image and a single box, return binary mask."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        box = box_xyxy[0].astype(np.float32)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False
        )
        mask = masks[0]

        # Morphological Close to fill holes (e.g. from flash/highlights)
        if morph_kernel_size > 0:
            mask_uint8 = (mask * 255).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            mask = mask_uint8 > 127
        else:
            if mask.dtype != np.bool_:
                mask = mask.astype(bool)
                
        return mask


def crop_with_black_bg(image_bgr, mask):
    """Apply mask to image, black background, crop to bounding box."""
    result = np.zeros_like(image_bgr)
    result[mask] = image_bgr[mask]

    # Crop to mask bounding box
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return result
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    return result[y1:y2, x1:x2]


# ==========================================
# 3. DINOv3 Semantic Segmentation (Inference)
# ==========================================
class FineTuneSegmentation(nn.Module):
    def __init__(self, model_name, num_classes, extract_layers):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.extract_layers = extract_layers

        self.embed_dim = self.backbone.config.hidden_size
        self.total_dim = self.embed_dim * len(extract_layers)

        self.head = nn.Sequential(
            nn.BatchNorm2d(self.total_dim),
            nn.Conv2d(self.total_dim, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        feats = []
        num_patches = (H // DINO_PATCH_SIZE) * (W // DINO_PATCH_SIZE)
        h_patch = H // DINO_PATCH_SIZE
        w_patch = W // DINO_PATCH_SIZE

        for layer_idx in self.extract_layers:
            feat = hidden_states[layer_idx]
            feat_spatial = feat[:, -num_patches:, :].permute(0, 2, 1)
            feat_spatial = feat_spatial.reshape(B, self.embed_dim, h_patch, w_patch)
            feats.append(feat_spatial)

        cat_feats = torch.cat(feats, dim=1)
        logits = self.head(cat_feats)
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)


def load_dino_model(device):
    """Load fine-tuned DINOv3 model for inference."""
    model = FineTuneSegmentation(
        DINO_MODEL_NAME, DINO_NUM_CLASSES, DINO_EXTRACT_LAYERS
    ).to(device)

    checkpoint = torch.load(DINO_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_mask(model, crop_bgr, device):
    """Run DINOv3 inference on a cropped flower image. Returns (H, W) uint8 class mask."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    orig_h, orig_w = crop_bgr.shape[:2]

    transform = v2.Compose([
        v2.Resize((DINO_IMG_SIZE, DINO_IMG_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # Resize back to original crop size
    pred_resized = cv2.resize(pred.astype('uint8'), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return pred_resized


# ==========================================
# 4. Visualization
# ==========================================
def decode_segmap(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS.items():
        rgb[mask == cls_idx] = color
    return rgb


def save_visualization(crop_bgr, pred_mask, save_path):
    """Save a 3-panel figure: crop / colored mask / overlay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seg_rgb = decode_segmap(pred_mask)
    overlay = cv2.addWeighted(crop_bgr[:, :, ::-1], 0.6, seg_rgb, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(crop_bgr[:, :, ::-1])
    axes[0].set_title("Cropped Flower")
    axes[0].axis('off')

    from matplotlib import colors as mcolors
    color_list = [np.array(CLASS_COLORS[i], dtype=np.float32) / 255.0 for i in range(len(CLASS_NAMES))]
    cmap = mcolors.ListedColormap(color_list)
    norm = mcolors.BoundaryNorm(np.arange(len(CLASS_NAMES) + 1), cmap.N)

    axes[1].imshow(pred_mask, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title("Pseudo Label (8-class)")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    patches = [
        plt.plot([], [], marker="s", ms=10, ls="", color=color_list[i], label=CLASS_NAMES[i])[0]
        for i in range(len(CLASS_NAMES))
    ]
    axes[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_pipeline_steps(img_bgr, boxes_all, primary_box, mask_binary, crop_bgr, pred_mask, save_path):
    """Save a 5-panel figure showing each pipeline stage:
    ① Original → ② YOLO+SPTS → ③ SAM2 → ④ Crop → ⑤ DINOv3
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    # [1] Original
    axes[0].imshow(img_bgr[:, :, ::-1])
    axes[0].set_title("① Original", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # [2] YOLO detections (gray) + SPTS primary (green)
    img_yolo = img_bgr.copy()
    if boxes_all is not None:
        for box in boxes_all:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (200, 200, 200), 2)
    if primary_box is not None:
        px1, py1, px2, py2 = primary_box[0].astype(int)
        cv2.rectangle(img_yolo, (px1, py1), (px2, py2), (0, 255, 0), 4)
    axes[1].imshow(img_yolo[:, :, ::-1])
    axes[1].set_title("② YOLO + SPTS", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # [3] SAM2 binary mask overlay
    img_sam = img_bgr.copy()
    sam_overlay = np.zeros_like(img_sam)
    sam_overlay[mask_binary] = [0, 255, 0]
    img_sam = cv2.addWeighted(img_sam, 0.6, sam_overlay, 0.4, 0)
    axes[2].imshow(img_sam[:, :, ::-1])
    axes[2].set_title("③ SAM2 Mask", fontsize=13, fontweight='bold')
    axes[2].axis('off')

    # [4] Cropped flower (black BG)
    axes[3].imshow(crop_bgr[:, :, ::-1])
    axes[3].set_title("④ Crop (Black BG)", fontsize=13, fontweight='bold')
    axes[3].axis('off')

    # [5] DINOv3 semantic segmentation overlay
    seg_rgb = decode_segmap(pred_mask)
    overlay_dino = cv2.addWeighted(crop_bgr[:, :, ::-1], 0.6, seg_rgb, 0.4, 0)
    axes[4].imshow(overlay_dino)
    axes[4].set_title("⑤ DINOv3 Pseudo Label", fontsize=13, fontweight='bold')
    axes[4].axis('off')

    # Legend
    color_list = [np.array(CLASS_COLORS[i], dtype=np.float32) / 255.0 for i in range(len(CLASS_NAMES))]
    patches = [
        plt.plot([], [], marker="s", ms=10, ls="", color=color_list[i], label=CLASS_NAMES[i])[0]
        for i in range(len(CLASS_NAMES))
    ]
    axes[4].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 5. Main Pipeline
# ==========================================
def main():
    print("=" * 60)
    print("  Pseudo-Labeling Pipeline")
    print("  YOLO → SPTS → SAM2 → DINOv3 → 8-class Mask")
    print("=" * 60)

    # Collect input images
    input_dir = Path(INPUT_DIR)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    print(f"\n[1/4] Found {len(img_paths)} images in {INPUT_DIR}")

    if len(img_paths) == 0:
        print("[!] No images found. Put raw photos in ./inputs/ and re-run.")
        return

    # Load models
    print(f"\n[2/4] Loading models on {DEVICE}...")
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_WEIGHTS)
    sam2 = SAM2Wrapper(DEVICE)
    dino_model = load_dino_model(DEVICE)
    print("  ✅ All models loaded.")

    # Process
    print(f"\n[3/4] Processing {len(img_paths)} images...")
    log_entries = []

    for img_path in tqdm(img_paths, desc="Pipeline"):
        stem = img_path.stem
        status = "success"
        n_det = 0

        boxes_all = None
        primary_box = None
        mask_binary = None
        crop = None
        pred_mask = None

        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                status = "read_error"
                raise ValueError("Cannot read image")

            h, w = img_bgr.shape[:2]

            # Stage 1: YOLO detection
            results = yolo_model.predict(
                source=str(img_path), conf=YOLO_CONF, iou=YOLO_IOU,
                imgsz=YOLO_IMG_SIZE, max_det=YOLO_MAX_DET, device=DEVICE, verbose=False
            )
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes_all = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
                n_det = len(boxes_all)
                primary_box = select_primary_flower(boxes_all, h, w)

            if primary_box is None:
                status = "no_detection"
                raise ValueError("No flower detected")

            # Stage 2: SAM2
            mask_binary = sam2.segment(img_bgr, primary_box)

            # Stage 3: Crop with black background
            crop = crop_with_black_bg(img_bgr, mask_binary)
            crop_path = os.path.join(CROP_DIR, f"{stem}.png")
            cv2.imwrite(crop_path, crop)

            # Stage 4: DINOv3 semantic segmentation
            pred_mask = predict_mask(dino_model, crop, DEVICE)

            # Save pseudo mask (uint8 class index)
            mask_path = os.path.join(MASK_DIR, f"{stem}.png")
            cv2.imwrite(mask_path, pred_mask)

            # Save 3-panel visualization
            viz_path = os.path.join(VIZ_DIR, f"{stem}_viz.png")
            save_visualization(crop, pred_mask, viz_path)

            # Save 5-panel pipeline steps
            steps_path = os.path.join(STEPS_DIR, f"{stem}_steps.png")
            save_pipeline_steps(img_bgr, boxes_all, primary_box, mask_binary, crop, pred_mask, steps_path)

        except Exception as e:
            if status == "success":
                status = f"error: {str(e)[:50]}"

        log_entries.append({
            "timestamp": datetime.now().isoformat(),
            "filename": img_path.name,
            "n_yolo_detections": n_det,
            "status": status
        })

    # Save log
    log_df = pd.DataFrame(log_entries)
    log_path = os.path.join(LOG_DIR, "pipeline_log.csv")
    log_df.to_csv(log_path, index=False)

    # Summary
    success = sum(1 for e in log_entries if e["status"] == "success")
    failed = len(log_entries) - success
    print(f"\n[4/4] Done!")
    print(f"  ✅ Success: {success}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Crops:   {CROP_DIR}")
    print(f"  📁 Masks:   {MASK_DIR}")
    print(f"  📁 Viz:     {VIZ_DIR}")
    print(f"  📁 Log:     {log_path}")


if __name__ == "__main__":
    main()
