"""
Baseline comparison: Traditional U-Net (From Scratch)
vs DINOv3 ViT-L for 8-class organ segmentation.

Purpose: Ablation study to demonstrate VFM backbone value against a standard, widely-used supervised baseline.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

# ==========================================
# 1. Config
# ==========================================
CONFIG = {
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    "dataset_dir": "./Datasets/v2_dataset_split",
    "save_dir": "./training_result/Finalexp_baseline_unet_resnet34_v1",
    "model_name": "U-Net (ResNet-34 encoder, ImageNet pretrained)",
    "img_size": 2400,        # same resolution as DINOv3 experiment
    "num_classes": 8,
    "epochs": 100,
    "batch_size": 2,
    # Differential LR: gentle on the pretrained ResNet-34 encoder, higher on the random decoder
    "lr_backbone": 1e-5,
    "lr_head": 5e-4,
    "num_workers": 1,
    "early_stopping_patience": 25,
    "early_stopping_min_delta": 5e-4,
    "use_checkpointing": True, # Required for 2400x2400 resolution
    "resume": False
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ==========================================
# 2. Dataset
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

        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        image = self.transforms(image)
        torch.manual_seed(seed)
        mask = self.mask_transforms(mask)
        return image, mask.squeeze(0)


# ==========================================
# 3. Model — U-Net with ImageNet-pretrained ResNet-34 encoder
#    (fair "conventional pretrained CNN" baseline vs the DINOv3 VFM)
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
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ResNetUNet(nn.Module):
    """U-Net decoder on an ImageNet-pretrained ResNet-34 encoder.
    Output logits are decoded to 1/2 resolution then bilinearly upsampled to the
    input size (keeps 2400x2400 memory tractable; gradient checkpointing on all stages)."""

    def __init__(self, in_channels=3, num_classes=8, pretrained=True, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet34(weights=weights)

        # --- Encoder (ResNet-34 stages) ---
        self.input_block = nn.Sequential(base.conv1, base.bn1, base.relu)  # 1/2,  64ch
        self.maxpool = base.maxpool                                        # 1/4
        self.layer1 = base.layer1   # 1/4,  64ch
        self.layer2 = base.layer2   # 1/8,  128ch
        self.layer3 = base.layer3   # 1/16, 256ch
        self.layer4 = base.layer4   # 1/32, 512ch

        # --- Decoder (transpose-conv up + skip concat + DoubleConv) ---
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

        s0 = cp(self.input_block, x)   # 1/2,  64
        p = self.maxpool(s0)           # 1/4
        s1 = cp(self.layer1, p)        # 1/4,  64
        s2 = cp(self.layer2, s1)       # 1/8,  128
        s3 = cp(self.layer3, s2)       # 1/16, 256
        b = cp(self.layer4, s3)        # 1/32, 512

        d4 = cp(self.dec4, torch.cat([self.up4(b), s3], dim=1))    # 1/16, 256
        d3 = cp(self.dec3, torch.cat([self.up3(d4), s2], dim=1))   # 1/8,  128
        d2 = cp(self.dec2, torch.cat([self.up2(d3), s1], dim=1))   # 1/4,  64
        d1 = cp(self.dec1, torch.cat([self.up1(d2), s0], dim=1))   # 1/2,  64

        logits = self.out_conv(d1)                                 # 1/2
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


# ==========================================
# 4. Metrics
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
# 5. Training loop
# ==========================================
def main():
    config_df = pd.DataFrame([CONFIG])
    config_df.to_csv(os.path.join(CONFIG["save_dir"], "params.csv"), index=False)
    print(f"✅ Training parameters saved to {CONFIG['save_dir']}/params.csv")

    training_logs = []
    log_path = os.path.join(CONFIG["save_dir"], "log.csv")

    train_dataset = SegmentationDataset(CONFIG["dataset_dir"], split="train", img_size=CONFIG["img_size"])
    val_dataset = SegmentationDataset(CONFIG["dataset_dir"], split="val", img_size=CONFIG["img_size"], is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    model = ResNetUNet(num_classes=CONFIG["num_classes"], pretrained=True, use_checkpointing=CONFIG["use_checkpointing"]).to(CONFIG["device"])
    print(f"Model: {CONFIG['model_name']}")

    # Differential LR: pretrained ResNet-34 encoder low, decoder high (mirrors DeepLabV3 baseline).
    encoder_prefixes = ("input_block", "maxpool", "layer1", "layer2", "layer3", "layer4")
    encoder_params = [p for n, p in model.named_parameters() if n.startswith(encoder_prefixes)]
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith(encoder_prefixes)]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": CONFIG["lr_backbone"]},
        {"params": decoder_params, "lr": CONFIG["lr_head"]},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = nn.CrossEntropyLoss()

    best_iou = 0.0
    no_improve_epochs = 0
    start_epoch = 0

    last_ckpt_path = os.path.join(CONFIG['save_dir'], "last.pth")
    if CONFIG.get("resume", False) and os.path.exists(last_ckpt_path):
        print(f"🔄 Resuming from {last_ckpt_path}...")
        checkpoint = torch.load(last_ckpt_path, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['best_iou']
        no_improve_epochs = checkpoint['no_improve_epochs']
        if os.path.exists(log_path):
            training_logs = pd.read_csv(log_path).to_dict('records')

    for epoch in range(start_epoch, CONFIG["epochs"]):
        # ================= Train =================
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{CONFIG['epochs']}]")

        for images, masks in pbar:
            images, masks = images.to(CONFIG["device"]), masks.to(CONFIG["device"])
            
            # Requires grad explicitly for checkpointing on the first layer input
            images.requires_grad = True 

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
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_unet.pth"))
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
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'no_improve_epochs': no_improve_epochs
        }, last_ckpt_path)

    print(f"\n{'='*50}")
    print(f"FINAL RESULT — U-Net Baseline (From Scratch)")
    print(f"Best val mIoU: {best_iou:.4f}")
    print(f"Log saved to: {log_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
