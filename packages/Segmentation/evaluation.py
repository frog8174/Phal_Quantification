import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import AutoModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import json

# ==========================================
# 1. 全局配置 (Config)
# ==========================================
CONFIG = {
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    
    # --- 模型設定 ---
    "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", 
    "img_size": 2400,            
    "patch_size": 16,
    "num_classes": 8,
    "extract_layers": [-1],      # MUST match fine-tuning.py (last layer)

    # --- Table 6: plain DINOv3 (ablation) on the held-out eval-dataset ---
    "checkpoint_path": "./training_result/Finalexp_v1/best_finetune_LandR.pth",
    "test_image_dir": "./Datasets/eval-dataset/images/",
    "test_mask_dir": "./Datasets/eval-dataset/masks/",
    "output_dir": "./Evaluation/Finalexp_v1_eval-dataset"
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

# ==========================================
# 2. 評估工具 (Evaluator Class) - 修復 NaN 計算問題
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
        label = self.num_classes * target_flat[mask].astype('int') + pred_flat[mask].astype('int')
        count = np.bincount(label, minlength=self.num_classes ** 2)
        conf_matrix = count.reshape(self.num_classes, self.num_classes)
        self.total_conf_matrix += conf_matrix

    def compute_metrics(self):
        cm = self.total_conf_matrix
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        
        # 【關鍵修復】：如果 union 為 0 (代表該類別根本沒出現在圖片中)，設定為 NaN，避免拉低 mIoU
        iou = np.where(union > 0, intersection / union, np.nan)
        class_acc = np.where(cm.sum(axis=1) > 0, intersection / cm.sum(axis=1), np.nan)
        
        pixel_acc = intersection.sum() / (cm.sum() + 1e-10)
        
        # nanmean 會自動忽略 NaN 的項目，只平均有出現的類別
        miou = np.nanmean(iou)
        macc = np.nanmean(class_acc)

        return {
            "Global Pixel Acc": float(pixel_acc),
            "mIoU": float(miou),
            "mAcc": float(macc),
            "IoU_per_class": {self.class_names[i]: float(iou[i] if not np.isnan(iou[i]) else 0.0) for i in range(self.num_classes)},
            "Acc_per_class": {self.class_names[i]: float(class_acc[i] if not np.isnan(class_acc[i]) else 0.0) for i in range(self.num_classes)}
        }

    def plot_confusion_matrix(self, save_path=None, normalize='true'):
        cm = self.total_conf_matrix.copy()
        if normalize == 'true':
            cm_to_plot = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
            title = 'Normalized Confusion Matrix (Recall)'
        else:
            cm_to_plot = cm
            title = 'Confusion Matrix (Counts)'

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_to_plot, annot=True, fmt='.2f' if normalize=='true' else '.0f', 
                    cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if save_path: plt.savefig(save_path)
        plt.close()

# ==========================================
# 3. 模型定義 (維持原樣)
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
# 4. 工具函數
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
# 5. 主流程
# ==========================================
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    global_evaluator = SegmentEvaluator(CONFIG['num_classes'], CLASS_NAMES)
    per_image_data = []

    print(f"Loading Fine-tuned Model from {CONFIG['checkpoint_path']}...")
    model = FineTuneSegmentation(CONFIG['model_name'], CONFIG['num_classes'], CONFIG['extract_layers']).to(CONFIG['device'])
    
    if os.path.exists(CONFIG["checkpoint_path"]):
        model.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"]))
    else:
        print(f"❌ Error: Checkpoint not found at {CONFIG['checkpoint_path']}!"); return

    model.eval()

    transform = v2.Compose([
        v2.Resize((CONFIG['img_size'], CONFIG['img_size']), interpolation=v2.InterpolationMode.BICUBIC),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_files = [f for f in os.listdir(CONFIG["test_image_dir"]) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Processing {len(test_files)} images...")

    for img_name in test_files:
        img_path = os.path.join(CONFIG["test_image_dir"], img_name)
        mask_path = os.path.join(CONFIG["test_mask_dir"], os.path.splitext(img_name)[0] + ".png")

        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        input_tensor = transform(img_pil).unsqueeze(0).to(CONFIG['device'])
        
        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype('uint8')
            pred_orig = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        if os.path.exists(mask_path):
            gt_mask = np.array(Image.open(mask_path))
            if len(gt_mask.shape) == 3: gt_mask = gt_mask[:,:,0]
            gt_orig = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 【防禦機制】：強制將未清理的 255 背景轉為 0
            if 255 in gt_orig:
                gt_orig[gt_orig == 255] = 0
            
            # 即時印出數值，讓你確認資料沒問題 (可隨時註解掉)
            print(f"[{img_name}] GT 包含的標籤: {np.unique(gt_orig)} | 模型預測的標籤: {np.unique(pred_orig)}")

            img_eval = SegmentEvaluator(CONFIG['num_classes'], CLASS_NAMES)
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
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=per_image_data[0].keys())
            writer.writeheader()
            writer.writerows(per_image_data)
        print(f"Detailed CSV report saved to {csv_path}")

    final_res = global_evaluator.compute_metrics()
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
        
    print("\n" + "="*30 + "\nFinal mIoU: {:.4f}\n".format(final_res['mIoU']) + "="*30)

if __name__ == "__main__":
    main()