"""
Fine-tuning (CoordConv-X variant)
=================================
與 fine-tuning.py 相同，唯一差別：在 segmentation head 前 concat 一個「絕對 x 座標」channel
(CoordConv，僅 x)，讓 head 取得「我在畫面多左/多右」的訊號，改善左右花瓣 (Petal_L/Petal_R)
的區分。

設計重點：
  - x 座標在 [-1, 1] 線性漸變，沿寬度方向 (w_patch)。
  - 座標是確定值，不能過 BatchNorm → 因此先對 backbone 特徵做 BN，再 concat 原始座標。
  - 前提：花朵經 YOLO 定位後大致置中、左右花瓣無大面積重疊，故「絕對 x」≈「相對中軸」，堪用。
  - 重疊場景仍救不了 → 必要時搭配 Column 中線後處理。

⚠️ 注意：本檔的模型架構與主線不同 (head 多吃 1 個 channel、BN 拆出)。
   用本檔訓練出來的 checkpoint，推論時 (inference.py / pipeline.py) 的 FineTuneSegmentation
   也要做同樣的 CoordConv 改動，否則 shape mismatch。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from transformers import AutoModel
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

# ==========================================
# 1. 全局配置
# ==========================================
PROJECT_LOC = 'end2end_coordconv_x'
CONFIG = {
    "device": "cuda:3" if torch.cuda.is_available() else "cpu",
    "dataset_dir": f"./Datasets/v2_dataset_split",
    "save_dir": f"./training_result/CoordConvX_v1",   # 獨立輸出，不蓋主線
    "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "local_weights_path": None,
    "img_size": 2400,
    "patch_size": 16,
    "num_classes": 8,
    "epochs": 100,
    "batch_size": 2,
    "lr_head": 0.0002,
    "lr_backbone": 0.000005,
    "num_workers": 1,
    "early_stopping_patience": 25,
    "early_stopping_min_delta": 5e-4,

    "extract_layers": [-1],
    "use_checkpointing": True,
    "resume": False
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. 數據集定義 (與主線一致，HFlip 已移除)
# ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=1600, is_train=True):
        self.root_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.mask_dir = os.path.join(self.root_dir, 'masks')
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])
        self.is_train = is_train
        self.img_size = img_size

        base_transforms = [
            v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        if is_train:
            self.transforms = v2.Compose([
                v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BICUBIC),
                # NOTE: HorizontalFlip 移除 — 會把左右花瓣搬位卻不換 6/7 標籤，破壞 L/R 訊號
                #       (CoordConv 依賴絕對 x，翻轉更會直接污染座標↔標籤對應)
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.mask_transforms = v2.Compose([
                v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.NEAREST),
                v2.RandomRotation(degrees=15),
                v2.ToImage(),
                v2.ToDtype(torch.long, scale=False)
            ])
        else:
            self.transforms = v2.Compose(base_transforms)
            self.mask_transforms = v2.Compose([
                v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.NEAREST),
                v2.ToImage(),
                v2.ToDtype(torch.long, scale=False)
            ])

    def __len__(self): return len(self.images)

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
# 3. 模型定義 (CoordConv-X)
# ==========================================
class FineTuneSegmentation(nn.Module):
    def __init__(self, model_name, num_classes, extract_layers, local_weights_path, use_checkpointing=False):
        super().__init__()

        print(f"Loading Backbone architecture: {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.extract_layers = extract_layers

        if local_weights_path is not None and os.path.exists(local_weights_path):
            print(f"Loading custom local weights from: {local_weights_path}...")
            checkpoint = torch.load(local_weights_path, map_location='cpu')

            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "student" in checkpoint:
                state_dict = checkpoint["student"]
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                elif k.startswith("backbone."):
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v

            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Local weights loaded! Message:\n{msg}")
        else:
            print(f"⚠️ Warning: Local weights not found at {local_weights_path}. Using HF pretrained weights instead.")

        for param in self.backbone.parameters():
            param.requires_grad = True

        if use_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.embed_dim = self.backbone.config.hidden_size
        self.total_dim = self.embed_dim * len(extract_layers)

        # --- CoordConv: BN 拆出，使座標 channel 不被 BatchNorm 洗掉 ---
        self.feat_bn = nn.BatchNorm2d(self.total_dim)
        # head 第一個 Conv 多吃 1 個 channel (絕對 x 座標)
        self.head = nn.Sequential(
            nn.Conv2d(self.total_dim + 1, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def _x_coord_channel(self, B, h_patch, w_patch, device, dtype):
        """產生 (B, 1, h_patch, w_patch) 的絕對 x 座標圖，沿寬度 -1 → 1。"""
        xs = torch.linspace(-1.0, 1.0, w_patch, device=device, dtype=dtype)
        return xs.view(1, 1, 1, w_patch).expand(B, 1, h_patch, w_patch)

    def forward(self, x):
        B, C, H, W = x.shape

        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        feats = []
        num_patches = (H // CONFIG['patch_size']) * (W // CONFIG['patch_size'])
        h_patch, w_patch = H // CONFIG['patch_size'], W // CONFIG['patch_size']

        for layer_idx in self.extract_layers:
            feat = hidden_states[layer_idx]
            feat_spatial = feat[:, -num_patches:, :].permute(0, 2, 1)
            feat_spatial = feat_spatial.reshape(B, self.embed_dim, h_patch, w_patch)
            feats.append(feat_spatial)

        cat_feats = torch.cat(feats, dim=1)

        # --- CoordConv: 先 BN 特徵，再 concat 原始 x 座標 ---
        feat = self.feat_bn(cat_feats)
        x_coord = self._x_coord_channel(B, h_patch, w_patch, feat.device, feat.dtype)
        feat = torch.cat([feat, x_coord], dim=1)

        logits = self.head(feat)
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

# ==========================================
# 4. 指標計算
# ==========================================
def calculate_iou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_list = []
    for cls in range(num_classes):
        p = (pred == cls); t = (target == cls)
        inter = (p & t).sum().item()
        union = p.sum().item() + t.sum().item() - inter
        if union > 0: iou_list.append(inter / union)
    return np.mean(iou_list) if iou_list else 0.0

# ==========================================
# 5. 主訓練流程
# ==========================================
def main():
    config_df = pd.DataFrame([CONFIG])
    config_df.to_csv(os.path.join(CONFIG['save_dir'], 'params.csv'), index=False)
    print(f"✅ Training parameters saved to {CONFIG['save_dir']}/params.csv")

    training_logs = []
    log_path = os.path.join(CONFIG['save_dir'], 'log.csv')

    train_dataset = SegmentationDataset(CONFIG['dataset_dir'], split='train', img_size=CONFIG['img_size'])
    val_dataset = SegmentationDataset(CONFIG['dataset_dir'], split='val', img_size=CONFIG['img_size'], is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    model = FineTuneSegmentation(
        CONFIG['model_name'],
        CONFIG['num_classes'],
        CONFIG['extract_layers'],
        CONFIG['local_weights_path'],
        CONFIG['use_checkpointing']
    ).to(CONFIG['device'])

    # --- 差異化學習率：backbone 一組，其餘 (feat_bn + head) 一組 ---
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CONFIG['lr_backbone']},
        {'params': head_params, 'lr': CONFIG['lr_head']}
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = nn.CrossEntropyLoss()

    best_iou = 0.0
    no_improve_epochs = 0
    start_epoch = 0
    early_stopping_patience = CONFIG["early_stopping_patience"]
    early_stopping_min_delta = CONFIG["early_stopping_min_delta"]

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

    for epoch in range(start_epoch, CONFIG['epochs']):
        # ================= Train Phase =================
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{CONFIG['epochs']}]")

        for images, masks in pbar:
            images, masks = images.to(CONFIG['device']), masks.to(CONFIG['device'])

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # ================= Val Phase =================
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(CONFIG['device']), masks.to(CONFIG['device'])
                logits = model(images)

                v_loss = criterion(logits, masks)
                val_loss += v_loss.item()
                val_iou += calculate_iou(logits, masks, CONFIG['num_classes'])

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_head = optimizer.param_groups[1]['lr']

        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] - T_Loss: {avg_train_loss:.4f}, V_Loss: {avg_val_loss:.4f}, V_mIoU: {avg_val_iou:.4f}")

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_miou": avg_val_iou,
            "lr_backbone": current_lr_backbone,
            "lr_head": current_lr_head,
            "best_miou": max(best_iou, avg_val_iou)
        }
        training_logs.append(log_entry)

        pd.DataFrame(training_logs).to_csv(log_path, index=False)

        if avg_val_iou > (best_iou + early_stopping_min_delta):
            best_iou = avg_val_iou
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_finetune_LandR.pth"))
            print(f"🔥 Best Model Saved with mIoU: {best_iou:.4f}")
        else:
            no_improve_epochs += 1
            print(
                f"EarlyStopping counter: {no_improve_epochs}/{early_stopping_patience} "
                f"(best mIoU: {best_iou:.4f}, min_delta: {early_stopping_min_delta})"
            )

        if no_improve_epochs >= early_stopping_patience:
            print(
                f"🛑 Early stopping triggered at epoch {epoch + 1}. "
                f"Best mIoU: {best_iou:.4f}"
            )
            break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'no_improve_epochs': no_improve_epochs
        }, last_ckpt_path)

if __name__ == "__main__":
    main()
