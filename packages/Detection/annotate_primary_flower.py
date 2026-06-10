"""
Primary Flower GT Annotation Tool
=================================
這是一個極速標註「主花 (Primary Flower)」Ground Truth 的工具。
它會動態執行 YOLO 抓出所有花朵的 Bounding Box，並讓你用滑鼠點擊選擇主花。

【操作方式】
- 滑鼠左鍵點擊：選擇你認為是主花的 Bounding Box (選中後會自動跳下一張)
- 按 's' 鍵：跳過這張圖 (例如這張圖沒有明顯的主花)
- 按 'b' 鍵：回上一張
- 按 'q' 或 ESC 鍵：儲存並退出

輸出檔案： primary_flower_gt.json
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ========== 設定區 ==========
# 自動抓取跨平台相對路徑
DETECTION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DETECTION_DIR.parent.parent

INPUT_DIR = DETECTION_DIR / "datasets" / "spts_testdataset"
OUTPUT_GT = DETECTION_DIR / "primary_flower_gt.json"
YOLO_WEIGHTS = DETECTION_DIR / "training_results" / "flower_detect" / "FD11" / "weights" / "best.pt"
DEVICE = 'cuda:3' # 這裡可依據你的顯卡改 'cuda:2' 或 '0'
# ============================

# 系統狀態
current_idx = 0
image_paths = []
gt_data = {}
current_boxes = []

def load_gt():
    global gt_data
    if os.path.exists(OUTPUT_GT):
        with open(OUTPUT_GT, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            print(f"Loaded {len(gt_data)} existing GT annotations.")

def save_gt():
    with open(OUTPUT_GT, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, indent=4)
    print(f"Saved {len(gt_data)} GT annotations to {OUTPUT_GT}")

def mouse_callback(event, x, y, flags, param):
    global current_idx, gt_data, current_boxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 檢查點擊在哪個 box 裡面
        # 如果重疊，選面積最小/最內層的那個
        selected_box = None
        min_area = float('inf')
        
        for box in current_boxes:
            bx1, by1, bx2, by2 = box
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                area = (bx2 - bx1) * (by2 - by1)
                if area < min_area:
                    min_area = area
                    selected_box = box.tolist()
        
        if selected_box is not None:
            # 記錄 GT
            img_name = image_paths[current_idx].name
            gt_data[img_name] = selected_box
            print(f"[{img_name}] Selected box: {selected_box}")
            
            # 畫個綠框給一點 feedback
            img_disp = param['disp'].copy()
            cv2.rectangle(img_disp, (int(selected_box[0]), int(selected_box[1])), 
                          (int(selected_box[2]), int(selected_box[3])), (0, 255, 0), 4)
            cv2.imshow("Annotator", img_disp)
            cv2.waitKey(150) # 停頓一下讓使用者看到綠框
            
            # 跳下一張
            current_idx += 1

def main():
    global current_idx, image_paths, current_boxes, gt_data
    
    print("Loading YOLO model...")
    model = YOLO(YOLO_WEIGHTS)
    if hasattr(model.model, 'to'):
        model.model.to(DEVICE)

    input_path = Path(INPUT_DIR)
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in exts])
    
    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    load_gt()

    # 從還沒標的圖片開始
    for i, p in enumerate(image_paths):
        if p.name not in gt_data:
            current_idx = i
            break

    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    
    while current_idx < len(image_paths):
        img_path = image_paths[current_idx]
        img = cv2.imread(str(img_path))
        if img is None:
            current_idx += 1
            continue
            
        # 1. 跑 YOLO 抓花朵
        results = model.predict(img, verbose=False, imgsz=960)
        current_boxes = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            current_boxes = results[0].boxes.xyxy.cpu().numpy()

        # 2. 畫 UI (紅框為候選)
        disp = img.copy()
        
        # 預先計算 spatial score 所需的參數
        h, w = img.shape[:2]
        img_cx, img_cy = w / 2.0, h / 2.0
        diag = np.sqrt(w**2 + h**2)
        
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in current_boxes] if len(current_boxes) > 0 else []
        max_area = max(areas) if len(areas) > 0 else 1.0

        for box in current_boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # 畫框線 (加粗)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # 計算 norm_area & norm_dist
            area = (x2 - x1) * (y2 - y1)
            norm_area = area / max_area
            
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
            norm_dist = dist / (diag / 2.0)
            
            # 準備文字 (放大字體適應高解析度)
            text = f"A:{norm_area:.2f} D:{norm_dist:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3.0
            thickness = 6
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 畫文字背景黑框，確保清楚
            text_y = max(y1 - 10, text_h + 10)
            cv2.rectangle(disp, (x1, text_y - text_h - 10), (x1 + text_w, text_y + 10), (0, 0, 0), -1)
            # 畫文字
            cv2.putText(disp, text, (x1, text_y), font, font_scale, (0, 255, 255), thickness)
            
        # 如果已經標註過了，畫綠框
        if img_path.name in gt_data:
            gt_box = gt_data[img_path.name]
            x1, y1, x2, y2 = map(int, gt_box)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 8)
            
            # 提示字眼也加個黑底 (放大適應高解析度)
            msg = "Already Annotated (Press Space to override/Next)"
            cv2.rectangle(disp, (40, 10), (2500, 120), (0, 0, 0), -1)
            cv2.putText(disp, msg, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 6)

        # 設定滑鼠回呼
        cv2.setMouseCallback("Annotator", mouse_callback, param={'disp': disp})
        
        cv2.imshow("Annotator", disp)
        
        # 等待按鍵
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27 or key == ord('q'): # ESC or q
            break
        elif key == ord('s'): # Skip
            print(f"[{img_path.name}] Skipped.")
            current_idx += 1
        elif key == ord('b'): # Back
            current_idx = max(0, current_idx - 1)
        elif key == 32: # Space (Next)
            current_idx += 1

    save_gt()
    cv2.destroyAllWindows()
    print("Annotation session finished.")

if __name__ == "__main__":
    main()
