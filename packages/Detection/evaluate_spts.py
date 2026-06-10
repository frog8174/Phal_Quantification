"""
Evaluate Primary Flower Identification (SPTS) Accuracy & Grid Search
====================================================================
讀取 GT 標註資料，測試 SPTS 演算法的準確率，並尋找最佳的 alpha / beta 參數。
"""

import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ========== 設定區 ==========
DETECTION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DETECTION_DIR.parent.parent
INPUT_DIR = DETECTION_DIR / "datasets" / "spts_testdataset"
GT_FILE = DETECTION_DIR / "primary_flower_gt.json"
YOLO_WEIGHTS = DETECTION_DIR / "training_results" / "flower_detect" / "FD11" / "weights" / "best.pt"
DEVICE = 'cuda:0'
# ============================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou_val

def sp_score(box, img_h, img_w, max_area, alpha, beta):
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    area = (box[2] - box[0]) * (box[3] - box[1])

    img_cx, img_cy = img_w / 2.0, img_h / 2.0
    diag = np.sqrt(img_w**2 + img_h**2)

    dist = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
    norm_area = area / max_area if max_area > 0 else 0
    norm_dist = dist / (diag / 2.0)

    return alpha * norm_area - beta * norm_dist

def select_primary(boxes, img_h, img_w, alpha, beta):
    if len(boxes) == 0:
        return None
    
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    max_area = max(areas)
    
    best_box = None
    best_score = -float('inf')
    
    for box in boxes:
        score = sp_score(box, img_h, img_w, max_area, alpha, beta)
        if score > best_score:
            best_score = score
            best_box = box
            
    return best_box

def main():
    if not GT_FILE.exists():
        print(f"GT file not found: {GT_FILE}")
        return

    with open(GT_FILE, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    print(f"Loaded {len(gt_data)} GT annotations.")

    print("Loading YOLO...")
    model = YOLO(YOLO_WEIGHTS)
    if hasattr(model.model, 'to'):
        model.model.to(DEVICE)

    print("Extracting Bounding Boxes and Confidence Scores from images...")
    image_boxes = {}
    image_confs = {}
    image_dims = {}

    for img_name in tqdm(gt_data.keys(), desc="YOLO Inference"):
        img_path = INPUT_DIR / img_name
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        image_dims[img_name] = (h, w)
        
        results = model.predict(img, verbose=False, imgsz=960)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            image_boxes[img_name] = results[0].boxes.xyxy.cpu().numpy()
            image_confs[img_name] = results[0].boxes.conf.cpu().numpy()
        else:
            image_boxes[img_name] = []
            image_confs[img_name] = []

    valid_imgs = list(image_dims.keys())
    print(f"Successfully processed {len(valid_imgs)} images with YOLO.")

    # 2. 測試預設參數
    print("\n--- Testing Default Parameters (alpha=1.0, beta=1.5) ---")
    correct = 0
    for img_name in valid_imgs:
        gt_box = gt_data[img_name]
        boxes = image_boxes[img_name]
        h, w = image_dims[img_name]
        
        pred_box = select_primary(boxes, h, w, alpha=1.0, beta=1.5)
        if pred_box is not None and iou(pred_box, gt_box) > 0.5:
            correct += 1
            
    acc = correct / len(valid_imgs) * 100
    print(f"Default Accuracy: {correct}/{len(valid_imgs)} ({acc:.2f}%)")

    # 3. 分析最佳參數 (alpha=2.0, beta=1.5) 的失敗案例，觀察 Confidence
    print("\n--- Failure Case Analysis (alpha=2.0, beta=1.5) ---")
    fail_count = 0
    for img_name in valid_imgs:
        gt_box = gt_data[img_name]
        boxes = image_boxes[img_name]
        confs = image_confs[img_name]
        h, w = image_dims[img_name]
        
        pred_box = select_primary(boxes, h, w, alpha=2.0, beta=1.5)
        
        if pred_box is None or iou(pred_box, gt_box) <= 0.5:
            fail_count += 1
            print(f"\n[Fail {fail_count}] Image: {img_name}")
            
            # 找出哪個 box 是 GT，哪個是 Pred
            gt_idx = -1
            pred_idx = -1
            for i, b in enumerate(boxes):
                if iou(b, gt_box) > 0.5:
                    gt_idx = i
                if pred_box is not None and np.array_equal(b, pred_box):
                    pred_idx = i
            
            # 輔助計算函數
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            max_area = max(areas) if areas else 1
            diag = np.sqrt(w**2 + h**2)
            
            def get_stats(idx):
                if idx == -1: return "Not Detected by YOLO"
                b = boxes[idx]
                c = confs[idx]
                area = (b[2]-b[0])*(b[3]-b[1])
                cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                return f"Conf: {c:.2f} | NormArea: {area/max_area:.2f} | NormDist: {dist/(diag/2):.2f}"

            print(f"  GT Box   -> {get_stats(gt_idx)}")
            print(f"  Pred Box -> {get_stats(pred_idx)}")
            
            # 判斷 YOLO Confidence 是否有幫助
            if gt_idx != -1 and pred_idx != -1:
                if confs[gt_idx] > confs[pred_idx]:
                    print("  💡 GT has HIGHER confidence! (Confidence could help here)")
                else:
                    print("  ⚠️ GT has LOWER or EQUAL confidence. (Confidence wouldn't help)")

    # 4. Grid Search 尋找最佳參數
    print("\n--- Starting Grid Search ---")
    alphas = np.arange(0.0, 3.1, 0.5)
    betas = np.arange(0.0, 3.1, 0.5)
    
    best_acc = 0
    best_params = []
    
    for a in alphas:
        for b in betas:
            correct = 0
            for img_name in valid_imgs:
                gt_box = gt_data[img_name]
                boxes = image_boxes[img_name]
                h, w = image_dims[img_name]
                
                pred_box = select_primary(boxes, h, w, alpha=a, beta=b)
                if pred_box is not None and iou(pred_box, gt_box) > 0.5:
                    correct += 1
            
            acc = correct / len(valid_imgs) * 100
            if acc > best_acc:
                best_acc = acc
                best_params = [(a, b)]
            elif acc == best_acc and acc > 0:
                best_params.append((a, b))
                
    print(f"Best Accuracy: {best_acc:.2f}%")
    print("Best Parameters (alpha, beta):")
    for p in best_params:
        print(f"  alpha={p[0]:.1f}, beta={p[1]:.1f}")

if __name__ == "__main__":
    main()
