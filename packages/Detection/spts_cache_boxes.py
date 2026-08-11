"""
Step 1 (需 ultralytics 環境):對 97 張 GT 影像跑一次 YOLO，把每張的所有花框
快取成 JSON。之後畫圖 / 算 grid 都讀這個快取，不必再跑 YOLO。

輸出: spts_eval_outputs/boxes_cache.json
"""
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm

D = Path(__file__).resolve().parent
INPUT_DIR = D / "datasets" / "spts_testdataset"
GT_FILE = D / "primary_flower_gt.json"
WEIGHTS = D / "training_results" / "flower_detect" / "FD11" / "weights" / "best.pt"
OUT = D / "spts_eval_outputs" / "boxes_cache.json"
DEVICE = "cuda:0"

gt = json.load(open(GT_FILE, encoding="utf-8"))
model = YOLO(str(WEIGHTS))
if hasattr(model.model, "to"):
    model.model.to(DEVICE)

cache = {}
for name in tqdm(gt.keys(), desc="YOLO"):
    p = INPUT_DIR / name
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    res = model.predict(img, verbose=False, imgsz=960)
    boxes = res[0].boxes.xyxy.cpu().numpy().tolist() if (res and res[0].boxes is not None) else []
    cache[name] = {"h": h, "w": w, "boxes": boxes}

OUT.parent.mkdir(exist_ok=True)
json.dump(cache, open(OUT, "w"))
print(f"Cached {len(cache)} images -> {OUT}")
