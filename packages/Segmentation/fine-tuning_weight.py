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
# 1. 全局配置 (Config)
# ==========================================
CONFIG = {
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "dataset_dir": f"./Datasets/dataset_cropped_19_5classes_LandR",
    "save_dir": f"./training_result/8classes_vit_DynamicFusion_v2",
    
    # 注意：ViT-L 通常是 24 層。如果你確定要 32 層，請確保模型是 ViT-H (dinov3-vith16)
    "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "local_weights_path": None,
    
    "img_size": 2400, 
    "patch_size": 16,
    "num_classes": 8,
    "epochs": 200,    
    "batch_size": 2,  
    "lr_head": 4e-4,  
    "lr_backbone": 5e-7, 
    "num_workers": 1,
    
    # 設定要融合的層數：這裡設定為 1 到 32 層
    # 如果執行時報錯 "IndexError: tuple index out of range"，請改為 list(range(1, 25))
    "extract_layers": list(range(1, 25)), 
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
        mask = self.mask_transforms(mask).squeeze(0)
        
        # 安全機制：強制把幽靈 255 背景轉為 0
        mask[mask == 255] = 0
        
        return image, mask

# ==========================================
# 3. 模型定義 (Dynamic Gated Fusion)
# ==========================================
class FineTuneSegmentation(nn.Module):
    def __init__(self, model_name, num_classes, extract_layers, local_weights_path, use_checkpointing=False):
        super().__init__()
        
        print(f"Loading Backbone architecture: {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.extract_layers = extract_layers
        
        # --- 載入本地權重 ---
        if local_weights_path is not None and os.path.exists(local_weights_path):
            print(f"Loading custom local weights from: {local_weights_path}...")
            checkpoint = torch.load(local_weights_path, map_location='cpu')
            if "model" in checkpoint: state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint: state_dict = checkpoint["state_dict"]
            elif "student" in checkpoint: state_dict = checkpoint["student"]
            else: state_dict = checkpoint 
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."): new_state_dict[k[7:]] = v
                elif k.startswith("backbone."): new_state_dict[k[9:]] = v
                else: new_state_dict[k] = v
            self.backbone.load_state_dict(new_state_dict, strict=False)
            print("✅ Local weights loaded successfully!")

        for param in self.backbone.parameters():
            param.requires_grad = True
          
        if use_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            
        self.embed_dim = self.backbone.config.hidden_size 
        
        # 🔥 建立可學習的權重矩陣 (對應每一層)，初始化為等權重 (1.0)
        self.layer_weights = nn.Parameter(torch.ones(len(self.extract_layers)))
        
        # 維度不變！因為我們是將各層特徵加權「疊加」在一起
        self.total_dim = self.embed_dim 

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
            
        num_patches = (H // CONFIG['patch_size']) * (W // CONFIG['patch_size'])
        h_patch, w_patch = H // CONFIG['patch_size'], W // CONFIG['patch_size']

        # 🔥 計算 Softmax，確保所有權重加起來為 100%
        normalized_weights = F.softmax(self.layer_weights, dim=0)
        
        fused_feat = 0.0  # 初始化融合特徵
        
        for i, layer_idx in enumerate(self.extract_layers):
            feat = hidden_states[layer_idx] 
            feat_spatial = feat[:, -num_patches:, :].permute(0, 2, 1)
            feat_spatial = feat_spatial.reshape(B, self.embed_dim, h_patch, w_patch)
            
            # 將該層特徵乘上其專屬權重並累加
            fused_feat = fused_feat + (normalized_weights[i] * feat_spatial)
            
        logits = self.head(fused_feat)
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

    training_logs = []
    log_path = os.path.join(CONFIG['save_dir'], 'log_with_weights.csv') # 更新檔名

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
    
    # --- 🔥 差異化學習率設定 (將 layer_weights 放入高學習率組) ---
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CONFIG['lr_backbone']},
        {'params': model.head.parameters(), 'lr': CONFIG['lr_head']},
        {'params': [model.layer_weights], 'lr': 0.05} # 讓它靈活更新
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = nn.CrossEntropyLoss(ignore_index=255) # 加上 ignore_index 防禦

    best_iou = 0.0
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
                
                v_loss = criterion(logits, masks)
                val_loss += v_loss.item()
                val_iou += calculate_iou(logits, masks, CONFIG['num_classes'])
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_head = optimizer.param_groups[1]['lr']
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] - T_Loss: {avg_train_loss:.4f}, V_Loss: {avg_val_loss:.4f}, V_mIoU: {avg_val_iou:.4f}")
        
        # ================= 🔥 權重記錄與日誌匯出 =================
        with torch.no_grad():
            # 取得目前的百分比權重分佈
            current_weights = F.softmax(model.layer_weights, dim=0).cpu().numpy()
            
            # 找出 Top 3 印在終端機
            top_3_indices = current_weights.argsort()[-3:][::-1]
            top_3_layers = [CONFIG['extract_layers'][i] for i in top_3_indices]
            top_3_probs = [current_weights[i] for i in top_3_indices]
            print(f"   📊 [融合權重] Top-3 關鍵層級: Layer {top_3_layers[0]} ({top_3_probs[0]:.1%}) | Layer {top_3_layers[1]} ({top_3_probs[1]:.1%}) | Layer {top_3_layers[2]} ({top_3_probs[2]:.1%})")

            # 建立日誌字典
            log_entry = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,    
                "val_miou": avg_val_iou,
                "lr_backbone": current_lr_backbone,
                "lr_head": current_lr_head,
                "best_miou": max(best_iou, avg_val_iou)
            }
            
            # 將每一層的權重單獨寫入欄位
            for i, layer_idx in enumerate(CONFIG['extract_layers']):
                log_entry[f"weight_layer_{layer_idx}"] = current_weights[i]
                
            training_logs.append(log_entry)
            pd.DataFrame(training_logs).to_csv(log_path, index=False)

        # ================= 模型儲存 =================
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], "best_finetune_model.pth"))
            print(f"   🔥 Best Model Saved! mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()