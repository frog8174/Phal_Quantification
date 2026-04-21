import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import AutoModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ==========================================
# 1. 設定區 (必須與 Fine-tuning 訓練時一致)
# ==========================================
PROJECT_LOC = './classifier'
CONFIG = {
    "device": "cuda:3" if torch.cuda.is_available() else "cpu",
    
    # 必須與訓練時一致 (DINOv3 Large)
    "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", 
    "img_size": 2400,           # 推論建議先用 1600，若 GPU 夠強可改 2400
    "patch_size": 16,
    "num_classes": 6, 
    "extract_layers": [-1], # 必須與 Fine-tuning 訓練時一致
    "local_weights_path" : "../models/pretrain/vit-l_97499.pth",
    
    # 權重路徑
    "checkpoint_path": "./end2end_5classes_i12_vitl_lastlayer/best_finetune.pth",
    
    # 測試與輸出路徑
    "test_image_dir":  "/mnt/nas2/Workspace/Aaron/Phal_process/manual_pick/raw_images",
    "output_dir": "./end2end_5classes_i12_vitl_lastlayer/inference_raw_images_P",
    "output_mask_dir": "./end2end_5classes_i12_vitl_lastlayer/output_masks"  # 新增：儲存純 Mask 的資料夾
}

CLASS_COLORS = {
    0: [0, 0, 0],       # 背景 (黑)
    1: [255, 105, 180], # 花瓣 (粉紅)
    2: [144, 238, 144], # 花萼上 (淺綠)
    3: [0, 100, 0],     # 花萼下 (深綠)
    4: [0, 0, 255],     # 唇瓣 (藍)
    5: [255, 255, 0]    # 蕊柱 (黃)
}
CLASS_NAMES = ["Background", "Petal", "Dorsal Sepal", "Lateral Sepal", "Labellum", "Column"]

# ==========================================
# 2. 模型定義 (改為 Fine-tuning 版本架構)
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
        # 推論時不需要 gradient checkpointing，直接 forward
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
            
        feats = []
        num_patches = (H // CONFIG['patch_size']) * (W // CONFIG['patch_size'])
        h_patch, w_patch = H // CONFIG['patch_size'], W // CONFIG['patch_size']

        for layer_idx in self.extract_layers:
            feat = hidden_states[layer_idx] 
            # 提取 Spatial tokens (排除 CLS/Registers)
            feat_spatial = feat[:, -num_patches:, :].permute(0, 2, 1)
            feat_spatial = feat_spatial.reshape(B, self.embed_dim, h_patch, w_patch)
            feats.append(feat_spatial)
            
        cat_feats = torch.cat(feats, dim=1)
        logits = self.head(cat_feats)
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

# ==========================================
# 3. 視覺化工具函數
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
# 4. 主推論流程
# ==========================================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["output_mask_dir"], exist_ok=True) # 建立存放純 Mask 的資料夾
    
    # A. 初始化模型
    model = FineTuneSegmentation(
        CONFIG['model_name'], 
        CONFIG['num_classes'], 
        CONFIG['extract_layers']
    ).to(CONFIG['device'])
    
    # B. 載入權重
    if os.path.exists(CONFIG["checkpoint_path"]):
        print(f"Loading Fine-tuned weights from {CONFIG['checkpoint_path']}")
        checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"])
        model.load_state_dict(checkpoint)
    else:
        print(f"Error: Checkpoint not found!")
        return

    model.eval()

    # C. 預處理
    transform = v2.Compose([
        v2.Resize((CONFIG['img_size'], CONFIG['img_size']), interpolation=v2.InterpolationMode.BICUBIC),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_files = [f for f in os.listdir(CONFIG["test_image_dir"]) if f.endswith(('.jpg', '.png', '.JPG'))]
    print(f"Total images: {len(test_files)}")

    for img_name in test_files:
        img_path = os.path.join(CONFIG["test_image_dir"], img_name)
        try:
            original_pil = Image.open(img_path).convert("RGB")
        except:
            continue
            
        orig_w, orig_h = original_pil.size
        
        # 1. 前處理
        input_tensor = transform(original_pil).unsqueeze(0).to(CONFIG['device'])
        
        # 2. 推論
        with torch.no_grad():
            logits = model(input_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        # 3. 後處理 (Resize 回原圖大小)
        pred_mask_orig = cv2.resize(pred_mask.astype('uint8'), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # ====================================================
        # 新增：單獨儲存純 Mask (存放 0~5 的灰階圖)
        # ====================================================
        mask_save_path = os.path.join(CONFIG["output_mask_dir"], f"{os.path.splitext(img_name)[0]}.png")
        mask_img = Image.fromarray(pred_mask_orig, mode='L') # mode='L' 代表 8-bit 灰階
        mask_img.save(mask_save_path)

        # 4. 視覺化 (RGB 與 Overlay)
        seg_rgb = decode_segmap(pred_mask_orig)
        blended = blend_images(original_pil, seg_rgb)
        
        # 5. 存檔與畫圖 (總覽圖)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_pil); axes[0].set_title("Original"); axes[0].axis('off')
        
        from matplotlib import colors
        color_list = [
            'black',              # 0
            '#FF69B4',            # 1 (Hot Pink)
            '#90EE90',            # 2 (Light Green)
            '#006400',            # 3 (Dark Green)
            'blue',               # 4
            'yellow'              # 5
        ]
        cmap = colors.ListedColormap(color_list)
        norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6], cmap.N)
        
        axes[1].imshow(pred_mask_orig, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1].set_title("Fine-tuned Prediction")
        axes[1].axis('off')
        
        axes[2].imshow(blended); axes[2].set_title("Overlay"); axes[2].axis('off')
        
        patches = [plt.plot([],[], marker="s", ms=10, ls="", color=np.array(CLASS_COLORS[i])/255.0, 
                    label=CLASS_NAMES[i])[0] for i in range(6)]
        axes[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], f"res_{img_name}"), dpi=150)
        plt.close()
        print(f"Processed: {img_name}")

if __name__ == "__main__":
    main()