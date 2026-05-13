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

# Output type:
# - Model output: segmentation logits with shape [B, 8, H, W]
# - Prediction output: class mask as a uint8 2D array after argmax, resized to the original image size
# - Visual output: RGB segmentation map, overlay image, and a saved 3-panel figure per input image

# ==========================================
# 1. 設定區 (必須與 Fine-tuning 訓練時一致)
# ==========================================
PROJECT_LOC = './classifier'
CONFIG = {
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    
    # 必須與訓練時一致 (DINOv3 Large)
    "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", 
    "img_size": 2400,           # 推論建議先用 1600，若 GPU 夠強可改 2400
    "patch_size": 16,
    "num_classes": 8, 
    "extract_layers": [-1], # 必須與 Fine-tuning 訓練時一致
    # 權重路徑 (指向你 Fine-tuning 產出的全模型權重)
    # "checkpoint_path": "./models_trained/best_finetune.pth",
    "checkpoint_path": "./training_result/7classes_vitl_layer-1_LandR_v3_argumented/best.pth",
    # 測試與輸出路徑
    "test_image_dir": "./Inference/test_datasets",
    "mask_output_dir": "./Inference/test_datasets_masks",
    "output_dir": "./Inference/7classes_vitl_layer-1_LandR_v3_argumented"
}

CLASS_COLORS = {
    0: [0, 0, 0],       # Background (背景 - 黑)
    1: [255, 255, 0],   # Column (蕊柱 - 黃)
    2: [0, 255, 0],     # Dorsal Sepal (上萼片/背萼片 - 綠)
    3: [0, 0, 255],     # Labellum (唇瓣 - 藍)
    4: [0, 255, 255],   # Lateral Sepal (側萼片 - 青色/淺藍)
    5: [255, 0, 0],     # Petal (花瓣 - 紅)
    6: [255, 0, 255],   # Petal_L (左花瓣 - 洋紅/紫)
    7: [255, 165, 0]    # Petal_R (右花瓣 - 橘色)
}

CLASS_NAMES = [
    "Background", "Column", "Dorsal Sepal", "Labellum", 
    "Lateral Sepal", "Petal", "Petal_L", "Petal_R"
]

COLOR_LIST = [
    np.array(CLASS_COLORS[idx], dtype=np.float32) / 255.0
    for idx in range(len(CLASS_NAMES))
]
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
    os.makedirs(CONFIG["mask_output_dir"], exist_ok=True)
    
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
            # argmax 取得類別
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        # 3. 後處理 (Resize 回原圖大小)
        pred_mask_orig = cv2.resize(pred_mask.astype('uint8'), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # 4. 視覺化
        seg_rgb = decode_segmap(pred_mask_orig)
        blended = blend_images(original_pil, seg_rgb)
        
        # 5. 存檔與畫圖
        # 單獨儲存純數值預測 Mask (0~7 類別)，檔名維持不變 (強制為 .png)，存入 mask_output_dir
        mask_filename = f"{os.path.splitext(img_name)[0]}.png"
        cv2.imwrite(os.path.join(CONFIG["mask_output_dir"], mask_filename), pred_mask_orig)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_pil); axes[0].set_title("Original"); axes[0].axis('off')
        
        from matplotlib import colors
        cmap = colors.ListedColormap(COLOR_LIST)
        norm = colors.BoundaryNorm(np.arange(len(CLASS_NAMES) + 1), cmap.N)
        
        axes[1].imshow(pred_mask_orig, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1].set_title("Fine-tuned Prediction")
        axes[1].axis('off')
        
        axes[2].imshow(blended); axes[2].set_title("Overlay"); axes[2].axis('off')
        
        patches = [
            plt.plot([], [], marker="s", ms=10, ls="", color=COLOR_LIST[i], label=CLASS_NAMES[i])[0]
            for i in range(len(CLASS_NAMES))
        ]
        
        # 調整圖例位置，避免擋到圖片
        axes[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], f"res_{img_name}"), dpi=150)
        plt.close()
        print(f"Processed: {img_name}")

if __name__ == "__main__":
    main()