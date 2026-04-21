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
import pandas as pd  # 新增 pandas 用於 CSV 操作

# ==========================================
# 1. 全局配置 (調整為 Fine-tuning 模式)
# ==========================================
PROJECT_LOC = 'end2end_5classes'
CONFIG = {
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    "dataset_dir": f"./Datasets/dataset_cropped_19_5classes_LandR",
    "save_dir": f"./training_result/7classes_vitl_layer-1_LandR_v3_argumented",
    "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    # "local_weights_path": "../models/pretrain/vit-l_97499.pth", # 你的本地權重路徑
    "local_weights_path": None,
    "img_size": 2400, 
    "patch_size": 16,
    "num_classes": 8,
    "epochs": 200,    
    "batch_size": 2,  
    "lr_head": 0.0004,  
    "lr_backbone": 0.000005, 
    "num_workers": 1,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 5e-4,
    
    "extract_layers": [-1], # len(feats)=32
    "use_checkpointing": True
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. 數據集定義
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
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.mask_transforms = v2.Compose([
                v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.NEAREST),
                v2.RandomHorizontalFlip(p=0.5), 
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
# 3. 模型定義 (修復本地權重載入)
# ==========================================
class FineTuneSegmentation(nn.Module):
    def __init__(self, model_name, num_classes, extract_layers, local_weights_path, use_checkpointing=False):
        super().__init__()
        
        # 1. 先載入 HF 預設的網路架構
        print(f"Loading Backbone architecture: {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.extract_layers = extract_layers
        
        # 2. 載入本地權重並覆蓋 Backbone
        if local_weights_path is not None and os.path.exists(local_weights_path):
            print(f"Loading custom local weights from: {local_weights_path}...")
            checkpoint = torch.load(local_weights_path, map_location='cpu')
            
            # 提取 state_dict
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "student" in checkpoint: # DINO 官方訓練腳本常將權重存於 student
                state_dict = checkpoint["student"]
            else:
                state_dict = checkpoint 
            
            # 清理 Key (處理 DDP 的 module. 前綴)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                elif k.startswith("backbone."): # 如果你的權重帶有 backbone. 前綴
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v
                    
            # 將整理好的權重載入到 backbone 裡 (修正了原本調用 model 的錯誤)
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Local weights loaded! Message:\n{msg}")
        else:
            print(f"⚠️ Warning: Local weights not found at {local_weights_path}. Using HF pretrained weights instead.")

        # --- 解凍參數 ---
        for param in self.backbone.parameters():
            param.requires_grad = True
          
        # --- 啟用 Gradient Checkpointing ---
        if use_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            
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
        num_patches = (H // CONFIG['patch_size']) * (W // CONFIG['patch_size'])
        h_patch, w_patch = H // CONFIG['patch_size'], W // CONFIG['patch_size']

        for layer_idx in self.extract_layers:
            feat = hidden_states[layer_idx] 
            feat_spatial = feat[:, -num_patches:, :].permute(0, 2, 1)
            feat_spatial = feat_spatial.reshape(B, self.embed_dim, h_patch, w_patch)
            feats.append(feat_spatial)
            
        cat_feats = torch.cat(feats, dim=1)
        logits = self.head(cat_feats)
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
    # --- 記錄訓練參數 ---
    config_df = pd.DataFrame([CONFIG])
    config_df.to_csv(os.path.join(CONFIG['save_dir'], 'params.csv'), index=False)
    print(f"✅ Training parameters saved to {CONFIG['save_dir']}/params.csv")

    # 初始化訓練日誌列表
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
    
    # --- 差異化學習率設定 ---
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CONFIG['lr_backbone']}, # 低學習率
        {'params': model.head.parameters(), 'lr': CONFIG['lr_head']}         # 高學習率
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = nn.CrossEntropyLoss()

    best_iou = 0.0
    no_improve_epochs = 0
    early_stopping_patience = CONFIG["early_stopping_patience"]
    early_stopping_min_delta = CONFIG["early_stopping_min_delta"]

    for epoch in range(CONFIG['epochs']):
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
                
                # 計算 val_loss 與 val_iou
                v_loss = criterion(logits, masks)
                val_loss += v_loss.item()
                val_iou += calculate_iou(logits, masks, CONFIG['num_classes'])
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_head = optimizer.param_groups[1]['lr']
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] - T_Loss: {avg_train_loss:.4f}, V_Loss: {avg_val_loss:.4f}, V_mIoU: {avg_val_iou:.4f}")
        
        # --- 記錄本輪指標 ---
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
        
        # 每一輪都覆蓋寫入一次
        pd.DataFrame(training_logs).to_csv(log_path, index=False)

        if avg_val_iou > (best_iou + early_stopping_min_delta):
            best_iou = avg_val_iou
            no_improve_epochs = 0
            # 儲存全模型參數
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

if __name__ == "__main__":
    main()
