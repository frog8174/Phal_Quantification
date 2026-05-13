import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO

# ===========================
# 使用者需設定的參數
# ===========================
YOLO_WEIGHTS = "training_results/flower_detect/FD11/weights/best.pt"
TEST_DIR = "datasets/flower/images"
OUTPUT_DIR = "spts_eval_outputs"
IMG_SIZE = 960
CONF_THRES = 0.5
IOU_THRES = 0.75
MAX_DET = 100
ALPHA = 1.0
BETA = 1.0

def process_and_visualize_spts():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = 2 if torch.cuda.is_available() else 'cpu'

    model = YOLO(YOLO_WEIGHTS)
    
    test_dir = Path(TEST_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir = out_dir / "visualizations"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        print(f"找不到測試資料夾: {test_dir}")
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [p for p in test_dir.iterdir() if p.suffix.lower() in exts]
    
    print(f"[INFO] 找到 {len(img_paths)} 張圖片，開始處理...")

    all_data = []

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        results = model.predict(
            source=str(img_path),
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMG_SIZE,
            max_det=MAX_DET,
            device=device,
            verbose=False
        )
        
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            print(f"[WARN] {img_path.name} 沒有偵測到任何目標")
            continue
            
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)

        # SPTS 運算
        img_cx, img_cy = w / 2.0, h / 2.0
        img_diag = np.sqrt(w ** 2 + h ** 2)

        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        max_area = areas.max()

        box_cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
        box_cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
        dists = np.sqrt((box_cx - img_cx) ** 2 + (box_cy - img_cy) ** 2)

        norm_area = areas / max_area
        norm_dist = dists / img_diag

        scores = ALPHA * norm_area - BETA * norm_dist
        best_idx = np.argmax(scores)
        
        sorted_indices = np.argsort(scores)[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(scores) + 1)

        # 紀錄數據
        for i in range(len(boxes_xyxy)):
            all_data.append({
                "Image": img_path.name,
                "Box_ID": i,
                "Rank": int(ranks[i]),
                "Is_Primary": bool(i == best_idx),
                "Score": float(scores[i]),
                "Norm_Area": float(norm_area[i]),
                "Norm_Dist": float(norm_dist[i]),
                "Box_X1": float(boxes_xyxy[i][0]),
                "Box_Y1": float(boxes_xyxy[i][1]),
                "Box_X2": float(boxes_xyxy[i][2]),
                "Box_Y2": float(boxes_xyxy[i][3])
            })

        # 開始繪圖 (只畫前幾張圖避免時間過長，或使用者需要的話全畫)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(img_rgb)
        ax.axis('off')
        
        ax.plot(img_cx, img_cy, 'r+', markersize=20, markeredgewidth=3, label='Image Center')

        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            box_w = x2 - x1
            box_h = y2 - y1
            
            is_best = (i == best_idx)
            color = '#ff0000' if is_best else '#00ff00'
            linewidth = 4 if is_best else 2
            linestyle = '-' if is_best else '--'
            
            rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=linewidth, edgecolor=color, facecolor='none', linestyle=linestyle)
            ax.add_patch(rect)
            
            ax.plot(box_cx[i], box_cy[i], marker='o', color=color, markersize=8)
            ax.plot([img_cx, box_cx[i]], [img_cy, box_cy[i]], color=color, linestyle=':', alpha=0.8)
            
            label = (f"R:{ranks[i]} | S:{scores[i]:.2f}\n"
                     f"A:{norm_area[i]:.2f} | D:{norm_dist[i]:.2f}")
            
            if is_best:
                label = "★ PRIMARY ★\n" + label
                
            ax.text(x1, y1-10, label, color='white', fontsize=12, fontweight='bold', 
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        plt.title(f"SPTS Evaluation - {img_path.name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(out_img_dir / f"{img_path.stem}_spts.jpg", dpi=150)
        plt.close(fig)

        print(f"[{img_path.name}] Done. Found {len(boxes_xyxy)} boxes. Primary: Box {best_idx}")

    # 儲存 CSV
    if all_data:
        df = pd.DataFrame(all_data)
        csv_path = out_dir / "spts_evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[DONE] 處理完畢！")
        print(f"總共處理了 {len(img_paths)} 張圖片。")
        print(f"數據已儲存至: {csv_path.absolute()}")
        print(f"視覺化圖片已儲存至: {out_img_dir.absolute()}")

if __name__ == "__main__":
    process_and_visualize_spts()
