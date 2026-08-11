"""
Evaluation script for the U-Net (from-scratch) baseline.
Mirrors evaluation.py / evaluation_deeplabv3.py exactly — same metrics, same outputs.
Model definition must match baseline_unet.py.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.models import resnet34, ResNet34_Weights
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
    "device": "cuda:3" if torch.cuda.is_available() else "cpu",
    "num_classes": 8,
    "img_size": 2400,

    # --- Table 6: U-Net baseline on the held-out eval-dataset ---
    "checkpoint_path": "./training_result/Finalexp_baseline_unet_resnet34_v1/best_unet.pth",
    "test_image_dir": "./Datasets/eval-dataset/images/",
    "test_mask_dir": "./Datasets/eval-dataset/masks/",
    "output_dir": "./Evaluation/baseline_unet_eval-dataset",
}

CLASS_COLORS = {
    0: [0, 0, 0], 1: [255, 255, 0], 2: [0, 255, 0], 3: [0, 0, 255],
    4: [0, 255, 255], 5: [255, 0, 0], 6: [255, 0, 255], 7: [255, 165, 0],
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
        self.total_conf_matrix += count.reshape(self.num_classes, self.num_classes)

    def compute_metrics(self):
        cm = self.total_conf_matrix
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = np.where(union > 0, intersection / union, np.nan)
        class_acc = np.where(cm.sum(axis=1) > 0, intersection / cm.sum(axis=1), np.nan)
        pixel_acc = intersection.sum() / (cm.sum() + 1e-10)
        return {
            "Global Pixel Acc": float(pixel_acc),
            "mIoU": float(np.nanmean(iou)),
            "mAcc": float(np.nanmean(class_acc)),
            "IoU_per_class": {self.class_names[i]: float(iou[i] if not np.isnan(iou[i]) else 0.0) for i in range(self.num_classes)},
            "Acc_per_class": {self.class_names[i]: float(class_acc[i] if not np.isnan(class_acc[i]) else 0.0) for i in range(self.num_classes)},
        }

    def plot_confusion_matrix(self, save_path=None, normalize="true"):
        cm = self.total_conf_matrix.copy()
        if normalize == "true":
            cm_to_plot = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)
            title = "Normalized Confusion Matrix (Recall) — U-Net Baseline"
        else:
            cm_to_plot = cm
            title = "Confusion Matrix (Counts) — U-Net Baseline"
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_to_plot, annot=True, fmt=".2f" if normalize == "true" else ".0f",
                    cmap="Greens", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ==========================================
# 3. Model — must match baseline_unet.py
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResNetUNet(nn.Module):
    """U-Net decoder on an ImageNet-pretrained ResNet-34 encoder. MUST match baseline_unet.py."""

    def __init__(self, in_channels=3, num_classes=8, pretrained=True, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet34(weights=weights)

        self.input_block = nn.Sequential(base.conv1, base.bn1, base.relu)  # 1/2,  64ch
        self.maxpool = base.maxpool                                        # 1/4
        self.layer1 = base.layer1   # 1/4,  64ch
        self.layer2 = base.layer2   # 1/8,  128ch
        self.layer3 = base.layer3   # 1/16, 256ch
        self.layer4 = base.layer4   # 1/32, 512ch

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256 + 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2:]

        def cp(module, *inp):
            if self.use_checkpointing and self.training:
                return torch.utils.checkpoint.checkpoint(module, *inp, use_reentrant=False)
            return module(*inp)

        s0 = cp(self.input_block, x)
        p = self.maxpool(s0)
        s1 = cp(self.layer1, p)
        s2 = cp(self.layer2, s1)
        s3 = cp(self.layer3, s2)
        b = cp(self.layer4, s3)

        d4 = cp(self.dec4, torch.cat([self.up4(b), s3], dim=1))
        d3 = cp(self.dec3, torch.cat([self.up3(d4), s2], dim=1))
        d2 = cp(self.dec2, torch.cat([self.up2(d3), s1], dim=1))
        d1 = cp(self.dec1, torch.cat([self.up1(d2), s0], dim=1))

        logits = self.out_conv(d1)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


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

    print(f"Loading U-Net from {CONFIG['checkpoint_path']}...")
    model = ResNetUNet(num_classes=CONFIG["num_classes"], pretrained=False, use_checkpointing=False).to(CONFIG["device"])
    if os.path.exists(CONFIG["checkpoint_path"]):
        model.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"]))
    else:
        print(f"❌ Error: Checkpoint not found at {CONFIG['checkpoint_path']}!")
        return
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

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

        seg_rgb = decode_segmap(pred_orig)
        blended = blend_images(img_pil, seg_rgb)
        save_name = os.path.join(CONFIG["output_dir"], f"res_{img_name}")
        cv2.imwrite(save_name, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    csv_path = os.path.join(CONFIG["output_dir"], "evaluation_report.csv")
    if per_image_data:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_data[0].keys())
            writer.writeheader()
            writer.writerows(per_image_data)
        print(f"Detailed CSV report saved to {csv_path}")

    final_res = global_evaluator.compute_metrics()
    final_res["model"] = "U-Net (from scratch)"
    final_res["total_params"] = total_params
    global_evaluator.plot_confusion_matrix(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"))

    # Raw confusion matrix (rows=True, cols=Pred) — exact L/R off-diagonal for the paper
    cm_path = os.path.join(CONFIG["output_dir"], "confusion_matrix.csv")
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["True\\Pred"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            w.writerow([name] + [int(x) for x in global_evaluator.total_conf_matrix[i]])
    print(f"Confusion matrix (counts) saved to {cm_path}")

    with open(os.path.join(CONFIG["output_dir"], "summary.json"), "w") as f:
        json.dump(final_res, f, indent=4)

    print("\n" + "=" * 50)
    print("U-Net Baseline Evaluation")
    print(f"Total params: {total_params:,}")
    print(f"Final mIoU: {final_res['mIoU']:.4f}")
    print(f"Global Pixel Acc: {final_res['Global Pixel Acc']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
