"""
Evaluation script for DeepLabV3-ResNet50 baseline.
Mirrors evaluation.py (DINOv3) exactly — same metrics, same outputs.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import json

# ==========================================
# 1. Config
# ==========================================
CONFIG = {
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",

    # --- Model ---
    "num_classes": 8,
    "img_size": 2400,

    # --- Paths (adjust to your actual checkpoint) ---
    "checkpoint_path": "./training_result/baseline_deeplabv3_resnet50/best.pth",
    "test_image_dir": "./Datasets/test_dataset/images/",
    "test_mask_dir": "./Datasets/test_dataset/masks/",
    "output_dir": "./Evaluation/baseline_deeplabv3_resnet50",
}

CLASS_COLORS = {
    0: [0, 0, 0],       # Background
    1: [255, 255, 0],   # Column
    2: [0, 255, 0],     # Dorsal Sepal
    3: [0, 0, 255],     # Labellum
    4: [0, 255, 255],   # Lateral Sepal
    5: [255, 0, 0],     # Petal
    6: [255, 0, 255],   # Petal_L
    7: [255, 165, 0],   # Petal_R
}

CLASS_NAMES = [
    "Background", "Column", "Dorsal Sepal", "Labellum",
    "Lateral Sepal", "Petal", "Petal_L", "Petal_R",
]


# ==========================================
# 2. Evaluator — identical to evaluation.py
# ==========================================
class SegmentEvaluator:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.total_conf_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, target):
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        mask = (target_flat >= 0) & (target_flat < self.num_classes)
        label = self.num_classes * target_flat[mask].astype("int") + pred_flat[mask].astype("int")
        count = np.bincount(label, minlength=self.num_classes ** 2)
        conf_matrix = count.reshape(self.num_classes, self.num_classes)
        self.total_conf_matrix += conf_matrix

    def compute_metrics(self):
        cm = self.total_conf_matrix
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection

        iou = np.where(union > 0, intersection / union, np.nan)
        class_acc = np.where(cm.sum(axis=1) > 0, intersection / cm.sum(axis=1), np.nan)

        pixel_acc = intersection.sum() / (cm.sum() + 1e-10)
        miou = np.nanmean(iou)
        macc = np.nanmean(class_acc)

        return {
            "Global Pixel Acc": float(pixel_acc),
            "mIoU": float(miou),
            "mAcc": float(macc),
            "IoU_per_class": {
                self.class_names[i]: float(iou[i] if not np.isnan(iou[i]) else 0.0)
                for i in range(self.num_classes)
            },
            "Acc_per_class": {
                self.class_names[i]: float(class_acc[i] if not np.isnan(class_acc[i]) else 0.0)
                for i in range(self.num_classes)
            },
        }

    def plot_confusion_matrix(self, save_path=None, normalize="true"):
        cm = self.total_conf_matrix.copy()
        if normalize == "true":
            cm_to_plot = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)
            title = "Normalized Confusion Matrix (Recall) — DeepLabV3 Baseline"
        else:
            cm_to_plot = cm
            title = "Confusion Matrix (Counts) — DeepLabV3 Baseline"

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_to_plot,
            annot=True,
            fmt=".2f" if normalize == "true" else ".0f",
            cmap="Oranges",  # Different colormap to distinguish from DINOv3 (Blues)
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ==========================================
# 3. Model — DeepLabV3 (must match baseline_deeplabv3.py)
# ==========================================
class DeepLabV3Baseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = deeplabv3_resnet50(
            weights="DeepLabV3_ResNet50_Weights.DEFAULT",
            weights_backbone="ResNet50_Weights.IMAGENET1K_V1",
        )
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        if self.model.aux_classifier is not None:
            aux_in = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(aux_in, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]


# ==========================================
# 4. Utilities
# ==========================================
def decode_segmap(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS.items():
        rgb[mask == cls_idx] = color
    return rgb


def blend_images(orig_img, seg_rgb, alpha=0.4):
    orig_img = np.array(orig_img)
    if orig_img.shape[:2] != seg_rgb.shape[:2]:
        seg_rgb = cv2.resize(seg_rgb, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(orig_img, 1 - alpha, seg_rgb, alpha, 0)


# ==========================================
# 5. Main
# ==========================================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    global_evaluator = SegmentEvaluator(CONFIG["num_classes"], CLASS_NAMES)
    per_image_data = []

    # --- Load model ---
    print(f"Loading DeepLabV3-ResNet50 from {CONFIG['checkpoint_path']}...")
    model = DeepLabV3Baseline(num_classes=CONFIG["num_classes"]).to(CONFIG["device"])

    if os.path.exists(CONFIG["checkpoint_path"]):
        model.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"]))
    else:
        print(f"❌ Error: Checkpoint not found at {CONFIG['checkpoint_path']}!")
        return

    model.eval()

    # --- Count params ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # --- Transform (same as training val transform) ---
    transform = v2.Compose([
        v2.Resize((CONFIG["img_size"], CONFIG["img_size"]), interpolation=v2.InterpolationMode.BICUBIC),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_files = [f for f in os.listdir(CONFIG["test_image_dir"]) if f.lower().endswith((".jpg", ".png"))]
    print(f"Processing {len(test_files)} images...")

    for img_name in test_files:
        img_path = os.path.join(CONFIG["test_image_dir"], img_name)
        mask_path = os.path.join(CONFIG["test_mask_dir"], os.path.splitext(img_name)[0] + ".png")

        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        input_tensor = transform(img_pil).unsqueeze(0).to(CONFIG["device"])

        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype("uint8")
            pred_orig = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        if os.path.exists(mask_path):
            gt_mask = np.array(Image.open(mask_path))
            if len(gt_mask.shape) == 3:
                gt_mask = gt_mask[:, :, 0]
            gt_orig = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Force-clean stray 255 values
            if 255 in gt_orig:
                gt_orig[gt_orig == 255] = 0

            print(f"[{img_name}] GT labels: {np.unique(gt_orig)} | Pred labels: {np.unique(pred_orig)}")

            img_eval = SegmentEvaluator(CONFIG["num_classes"], CLASS_NAMES)
            img_eval.update(pred_orig, gt_orig)
            res = img_eval.compute_metrics()

            row = {"FileName": img_name, "mIoU": f"{res['mIoU']:.4f}", "PixelAcc": f"{res['Global Pixel Acc']:.4f}"}
            for cls in CLASS_NAMES:
                row[f"IoU_{cls}"] = f"{res['IoU_per_class'][cls]:.4f}"
                row[f"Acc_{cls}"] = f"{res['Acc_per_class'][cls]:.4f}"
            per_image_data.append(row)

            global_evaluator.update(pred_orig, gt_orig)

        # Save overlay
        seg_rgb = decode_segmap(pred_orig)
        blended = blend_images(img_pil, seg_rgb)
        save_name = os.path.join(CONFIG["output_dir"], f"res_{img_name}")
        cv2.imwrite(save_name, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    # --- Save CSV report ---
    csv_path = os.path.join(CONFIG["output_dir"], "evaluation_report.csv")
    if per_image_data:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_data[0].keys())
            writer.writeheader()
            writer.writerows(per_image_data)
        print(f"Detailed CSV report saved to {csv_path}")

    # --- Save confusion matrix + summary ---
    final_res = global_evaluator.compute_metrics()
    final_res["model"] = "DeepLabV3-ResNet50"
    final_res["total_params"] = total_params
    final_res["trainable_params"] = trainable_params

    global_evaluator.plot_confusion_matrix(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"))

    with open(os.path.join(CONFIG["output_dir"], "summary.json"), "w") as f:
        json.dump(final_res, f, indent=4)

    print(f"\n{'='*50}")
    print(f"DeepLabV3-ResNet50 Baseline Evaluation")
    print(f"Total params: {total_params:,}")
    print(f"Final mIoU: {final_res['mIoU']:.4f}")
    print(f"Global Pixel Acc: {final_res['Global Pixel Acc']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
