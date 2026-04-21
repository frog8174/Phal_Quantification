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

print("predict.py is runing...")
# ==========================================
# 1. 設定區 (必須與 Fine-tuning 訓練時一致)
# ==========================================
PROJECT_LOC = './classifier'
CONFIG = {
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    
    # 必須與訓練時一致 (DINOv3 Large)
    "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", 
    "img_size": 1600,           # 推論建議先用 1600，若 GPU 夠強可改 2400
    "patch_size": 16,
    "num_classes": 5, 
    "extract_layers": [-1], # 必須與 Fine-tuning 訓練時一致
    
    # 權重路徑 (指向你 Fine-tuning 產出的全模型權重)
    "checkpoint_path": "./models_trained/best_finetune.pth",
    
    # 測試與輸出路徑
    "test_image_dir": "/mnt/nas2/Workspace/Aaron/Phal_process/manual_pick/sam_crops",
    "output_dir": "/mnt/nas2/Workspace/Aaron/Phal_process/manual_pick/fine-tuned-l16-last"
}

CLASS_COLORS = {
    0: [0, 0, 0],       # 背景
    1: [255, 0, 0],     # 花瓣
    2: [0, 255, 0],     # 花萼
    3: [0, 0, 255],     # 唇瓣
    4: [255, 255, 0]    # 蕊柱
}
CLASS_NAMES = ["Background", "Petal", "Sepal", "Labellum", "Column"]

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
# 3. 視覺化工具函數 (保持不變)
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
    
    # A. 初始化模型
    # 注意：這裡使用 FineTuneSegmentation 而非 LinearProbeSegmentation
    model = FineTuneSegmentation(
        CONFIG['model_name'], 
        CONFIG['num_classes'], 
        CONFIG['extract_layers']
    ).to(CONFIG['device'])
    
    # B. 載入權重 (Fine-tuning 時存的是全模型 state_dict)
    if os.path.exists(CONFIG["checkpoint_path"]):
        print(f"Loading Fine-tuned weights from {CONFIG['checkpoint_path']}")
        checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"])
        # 直接載入全模型
        model.load_state_dict(checkpoint)
        print(f"Good: Checkpoint Loaded!")
    else:
        print(f"Error: Checkpoint not found!")
        return

    model.eval()

    # C. 預處理 (針對 NumPy 1.x/2.x 相容性優化)
    transform = v2.Compose([
        v2.Resize((CONFIG['img_size'], CONFIG['img_size']), interpolation=v2.InterpolationMode.BICUBIC),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_files = [f for f in os.listdir(CONFIG["test_image_dir"]) if f.endswith(('.jpg', '.png'))]
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
            # argmax 取得類別
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        # 3. 後處理 (Resize 回原圖大小)
        pred_mask_orig = cv2.resize(pred_mask.astype('uint8'), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # 4. 視覺化
        seg_rgb = decode_segmap(pred_mask_orig)
        blended = blend_images(original_pil, seg_rgb)
        
        # 5. 存檔與畫圖
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_pil); axes[0].set_title("Original"); axes[0].axis('off')
        
        from matplotlib import colors
        cmap = colors.ListedColormap(['black', 'red', 'green', 'blue', 'yellow'])
        norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)
        
        axes[1].imshow(pred_mask_orig, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1].set_title("Fine-tuned Prediction")
        axes[1].axis('off')
        
        axes[2].imshow(blended); axes[2].set_title("Overlay"); axes[2].axis('off')
        
        # Legend
        patches = [plt.plot([],[], marker="s", ms=10, ls="", color=CLASS_COLORS[i]/np.array(255), 
                    label=CLASS_NAMES[i])[0] for i in range(5)]
        axes[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], f"res_{img_name}"), dpi=150)
        plt.close()
        print(f"Processed: {img_name}")

if __name__ == "__main__":
    main()