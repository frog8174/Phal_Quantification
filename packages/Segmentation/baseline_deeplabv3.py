"""
Baseline comparison: DeepLabV3 (ResNet-50, ImageNet pretrained)
vs DINOv3 ViT-L / CoordConv-X for 8-class organ segmentation.

Table 6 setup: train on v2_dataset_split (20/4), test on eval-dataset (9, held-out).
  - Same dataset split, same augmentation (HFlip removed), same loss, same evaluation
  - Difference vs proposed: backbone (ResNet-50 supervised vs DINOv3 self-supervised)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

# ==========================================
# 1. Config — mirrors fine-tuning.py exactly
# ==========================================
CONFIG = {
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    "dataset_dir": "./Datasets/v2_dataset_split",
    "save_dir": "./training_result/Finalexp_baseline_deeplabv3_v1",
    "model_name": "DeepLabV3-ResNet50 (ImageNet pretrained)",
    "img_size": 2400,        # same resolution as DINOv3 experiment
    "num_classes": 8,
    "epochs": 100,
    "batch_size": 2,
    "lr": 1e-4,              # single LR (no differential — standard practice for DeepLab)
    "lr_backbone": 1e-5,     # fine-tune backbone at lower LR
    "lr_head": 1e-4,         # decoder head at higher LR
    "num_workers": 1,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 5e-4,
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ==========================================
# 2. Dataset — identical to fine-tuning.py
# ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=2400, is_train=True):
        self.root_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.root_dir, "images")
        self.mask_dir = os.path.join(self.root_dir, "masks")
        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith((".jpg", ".png")) and f != "Thumbs.db"
        ])
        self.is_train = is_train
        self.img_size = img_size

        if is_train:
            self.transforms = v2.Compose([
                v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BICUBIC),
                # NOTE: HorizontalFlip removed — it moves L/R petals without swapping labels 6/7,
                #       corrupting the L/R signal. Matches fine-tuning.py / CoordConv for a fair comparison.
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.mask_transforms = v2.Compose([
                v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.NEAREST),
                # NOTE: HorizontalFlip removed — it moves L/R petals without swapping labels 6/7,
                #       corrupting the L/R signal. Matches fine-tuning.py / CoordConv for a fair comparison.
                v2.RandomRotation(degrees=15),
                v2.ToImage(),
                v2.ToDtype(torch.long, scale=False),
            ])
        else:
            self.transforms = v2.Compose([
                v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.mask_transforms = v2.Compose([
                v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.NEAREST),
                v2.ToImage(),
                v2.ToDtype(torch.long, scale=False),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = os.path.splitext(img_name)[0] + ".png"
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")

        # Synchronized augmentation (same seed for image & mask)
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        image = self.transforms(image)
        torch.manual_seed(seed)
        mask = self.mask_transforms(mask)
        return image, mask.squeeze(0)


# ==========================================
# 3. Model — DeepLabV3 with ResNet-50
# ==========================================
class DeepLabV3Baseline(nn.Module):
    """
    torchvision DeepLabV3 with ResNet-50 backbone.
    Replaces the final classifier head to output num_classes channels.
    Uses gradient checkpointing on backbone for memory savings at 2400px.
    """

    def __init__(self, num_classes, pretrained_backbone=True):
        super().__init__()
        # Load pretrained DeepLabV3 (COCO weights for backbone + ASPP decoder)
        self.model = deeplabv3_resnet50(
            weights="DeepLabV3_ResNet50_Weights.DEFAULT",
            weights_backbone="ResNet50_Weights.IMAGENET1K_V1" if pretrained_backbone else None,
        )
        # Replace classifier head: 21 (COCO) → num_classes
        in_channels = self.model.classifier[4].in_channels  # 256
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # Also replace aux classifier if present
        if self.model.aux_classifier is not None:
            aux_in = self.model.aux_classifier[4].in_channels  # 256
            self.model.aux_classifier[4] = nn.Conv2d(aux_in, num_classes, kernel_size=1)

        # Enable gradient checkpointing on backbone for VRAM savings
        if hasattr(self.model.backbone, "layer1"):
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                layer = getattr(self.model.backbone, layer_name)
                layer.register_forward_hook(lambda m, i, o: o)
                # Use torch.utils.checkpoint for each ResNet block
                for block in layer:
                    block._original_forward = block.forward
                    block.forward = lambda *args, _block=block: torch.utils.checkpoint.checkpoint(
                        _block._original_forward, *args, use_reentrant=False
                    )

    def forward(self, x):
        out = self.model(x)
        return out["out"]  # (B, num_classes, H, W) — already input-resolution


# ==========================================
# 4. Metrics — identical to fine-tuning.py
# ==========================================
def calculate_iou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_list = []
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().item()
        union = p.sum().item() + t.sum().item() - inter
        if union > 0:
            iou_list.append(inter / union)
    return np.mean(iou_list) if iou_list else 0.0


# ==========================================
# 5. Training loop — mirrors fine-tuning.py
# ==========================================
def main():
    # Save config
    config_df = pd.DataFrame([CONFIG])
    config_df.to_csv(os.path.join(CONFIG["save_dir"], "params.csv"), index=False)
    print(f"✅ Training parameters saved to {CONFIG['save_dir']}/params.csv")

    training_logs = []
    log_path = os.path.join(CONFIG["save_dir"], "log.csv")

    # Dataset & DataLoader
    train_dataset = SegmentationDataset(CONFIG["dataset_dir"], split="train", img_size=CONFIG["img_size"])
    val_dataset = SegmentationDataset(CONFIG["dataset_dir"], split="val", img_size=CONFIG["img_size"], is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # Model
    model = DeepLabV3Baseline(num_classes=CONFIG["num_classes"]).to(CONFIG["device"])
    print(f"Model: {CONFIG['model_name']}")

    # Optimizer — differential LR (same strategy as DINOv3 experiment)
    backbone_params = list(model.model.backbone.parameters())
    head_params = (
        list(model.model.classifier.parameters())
        + (list(model.model.aux_classifier.parameters()) if model.model.aux_classifier else [])
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": CONFIG["lr_backbone"]},
            {"params": head_params, "lr": CONFIG["lr_head"]},
        ],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = nn.CrossEntropyLoss()

    best_iou = 0.0
    no_improve_epochs = 0

    for epoch in range(CONFIG["epochs"]):
        # ================= Train =================
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{CONFIG['epochs']}]")

        for images, masks in pbar:
            images, masks = images.to(CONFIG["device"]), masks.to(CONFIG["device"])

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # ================= Val =================
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(CONFIG["device"]), masks.to(CONFIG["device"])
                logits = model(images)
                v_loss = criterion(logits, masks)
                val_loss += v_loss.item()
                val_iou += calculate_iou(logits, masks, CONFIG["num_classes"])

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        print(
            f"Epoch [{epoch + 1}/{CONFIG['epochs']}] - "
            f"T_Loss: {avg_train_loss:.4f}, V_Loss: {avg_val_loss:.4f}, V_mIoU: {avg_val_iou:.4f}"
        )

        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_miou": avg_val_iou,
            "lr_backbone": current_lr_backbone,
            "lr_head": current_lr_head,
            "best_miou": max(best_iou, avg_val_iou),
        }
        training_logs.append(log_entry)
        pd.DataFrame(training_logs).to_csv(log_path, index=False)

        # Checkpoint
        if avg_val_iou > (best_iou + CONFIG["early_stopping_min_delta"]):
            best_iou = avg_val_iou
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best.pth"))
            print(f"🔥 Best Model Saved with mIoU: {best_iou:.4f}")
        else:
            no_improve_epochs += 1
            print(
                f"EarlyStopping counter: {no_improve_epochs}/{CONFIG['early_stopping_patience']} "
                f"(best mIoU: {best_iou:.4f})"
            )

        if no_improve_epochs >= CONFIG["early_stopping_patience"]:
            print(f"🛑 Early stopping at epoch {epoch + 1}. Best mIoU: {best_iou:.4f}")
            break

    print(f"\n{'='*50}")
    print(f"FINAL RESULT — DeepLabV3 ResNet-50 Baseline")
    print(f"Best val mIoU: {best_iou:.4f}")
    print(f"Log saved to: {log_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
