"""
Baseline comparison: Traditional U-Net (From Scratch)
vs DINOv3 ViT-L for 8-class organ segmentation.

Purpose: Ablation study to demonstrate VFM backbone value against a standard, widely-used supervised baseline.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

# ==========================================
# 1. Config
# ==========================================
CONFIG = {
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "dataset_dir": "./Datasets/v2_dataset_split",
    "save_dir": "./training_result/baseline_unet",
    "model_name": "U-Net-lr2e-4 (Trained from scratch)",
    "img_size": 2400,        # same resolution as DINOv3 experiment
    "num_classes": 8,
    "epochs": 200,
    "batch_size": 2,
    "lr": 2e-4,              # standard U-Net LR
    "num_workers": 1,
    "early_stopping_patience": 200,
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
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.mask_transforms = v2.Compose([
                v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.NEAREST),
                v2.RandomHorizontalFlip(p=0.5),
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
# 3. Model — Standard U-Net (From Scratch)
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

class UNetBaseline(nn.Module):
    def __init__(self, in_channels=3, num_classes=8, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 為了在高解析度下節省 VRAM，使用 Gradient Checkpointing
        def cp(module, x):
            if self.use_checkpointing and x.requires_grad:
                return torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)
            return module(x)
            
        x1 = cp(self.down1, x)
        x2 = cp(self.down2, self.pool1(x1))
        x3 = cp(self.down3, self.pool2(x2))
        x4 = cp(self.down4, self.pool3(x3))
        
        x5 = cp(self.bottleneck, self.pool4(x4))
        
        x = self.up4(x5)
        x = cp(self.up_conv4, torch.cat([x, x4], dim=1))
        
        x = self.up3(x)
        x = cp(self.up_conv3, torch.cat([x, x3], dim=1))
        
        x = self.up2(x)
        x = cp(self.up_conv2, torch.cat([x, x2], dim=1))
        
        x = self.up1(x)
        x = cp(self.up_conv1, torch.cat([x, x1], dim=1))
        
        return self.out_conv(x)


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

    model = UNetBaseline(num_classes=CONFIG["num_classes"], use_checkpointing=CONFIG["use_checkpointing"]).to(CONFIG["device"])
    print(f"Model: {CONFIG['model_name']}")

    # U-Net trains from scratch, so single learning rate is used.
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
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

        current_lr = optimizer.param_groups[0]["lr"]

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
            "lr_backbone": current_lr, # unified LR
            "lr_head": current_lr,     # unified LR
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
