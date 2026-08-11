"""
比較 SPTS 在 default (alpha=1.0, beta=1.5) 與 optimal (alpha=2.0, beta=1.5) 下
選到的主花有何不同，並對照 GT 判斷每個變動是 FIXED / BROKE / 仍錯 / 仍對。

輸出:
  spts_eval_outputs/spts_default_vs_optimal.csv  (每張 GT 圖一列)
  終端印出「選擇改變」的影像清單 + verdict。

用法(在有 ultralytics 的環境):
  python evaluate_spts_diff.py
"""
import json
import csv
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

DETECTION_DIR = Path(__file__).resolve().parent
INPUT_DIR = DETECTION_DIR / "datasets" / "spts_testdataset"
GT_FILE = DETECTION_DIR / "primary_flower_gt.json"
YOLO_WEIGHTS = DETECTION_DIR / "training_results" / "flower_detect" / "FD11" / "weights" / "best.pt"
OUT_CSV = DETECTION_DIR / "spts_eval_outputs" / "spts_default_vs_optimal.csv"
VIZ_DIR = DETECTION_DIR / "spts_eval_outputs" / "default_vs_optimal_viz"
DEVICE = "cuda:0"
DEFAULT = (1.0, 1.5)
OPTIMAL = (2.0, 1.5)
SAVE_VIZ = True          # 對「選擇改變」的影像輸出左右對比圖
PANEL_W = 800            # 每個面板縮放後寬度(px)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def sp_score(box, h, w, max_area, alpha, beta):
    bw, bh = box[2]-box[0], box[3]-box[1]
    norm_area = (bw*bh) / max_area if max_area > 0 else 0
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    dist = np.hypot(cx - w/2, cy - h/2)
    norm_dist = dist / np.hypot(w/2, h/2)
    return alpha*norm_area - beta*norm_dist


def _panel(img, boxes, sel_idx, gt, header, sel_color):
    """縮放影像、畫所有框(灰)+ GT(藍)+ 選中框(sel_color)+ 頂部標題列。"""
    h, w = img.shape[:2]
    sc = PANEL_W / w
    vis = cv2.resize(img, (PANEL_W, int(h * sc)))
    def R(b, color, th):
        cv2.rectangle(vis, (int(b[0]*sc), int(b[1]*sc)),
                      (int(b[2]*sc), int(b[3]*sc)), color, th)
    for b in boxes:                       # 所有偵測框 = 細灰
        R(b, (150, 150, 150), 1)
    R(gt, (255, 120, 0), 2)               # GT = 藍(BGR)
    R(boxes[sel_idx], sel_color, 3)       # 選中主花 = 粗(紅/綠)
    bar = np.full((34, PANEL_W, 3), 40, np.uint8)
    cv2.putText(bar, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def save_single(img, boxes, sel_idx, gt, name, is_correct):
    """optimal 結果的單面板圖 -> correct/ 或 wrong/。選中框:對=綠、錯=紅。"""
    tag = "correct" if is_correct else "wrong"
    color = (0, 200, 0) if is_correct else (0, 0, 255)
    header = f"Optimal (a=2.0, b=1.5) - {tag.upper()}   blue=GT"
    panel = _panel(img, boxes, sel_idx, gt, header, color)
    d = VIZ_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{Path(name).stem}.jpg"), panel)


def save_compare(img, boxes, di, oi, gt, name, verdict):
    """default vs optimal 左右對比 -> changed/。"""
    left = _panel(img, boxes, di, gt, "Default  (a=1.0, b=1.5)", (0, 0, 255))
    right = _panel(img, boxes, oi, gt, "Optimal  (a=2.0, b=1.5)", (0, 200, 0))
    hh = min(left.shape[0], right.shape[0])
    combo = np.hstack([left[:hh], np.full((hh, 6, 3), 255, np.uint8), right[:hh]])
    title = np.full((34, combo.shape[1], 3), 20, np.uint8)
    txt = f"{name}   [{verdict}]   blue=GT  red=default pick  green=optimal pick"
    cv2.putText(title, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    d = VIZ_DIR / "changed"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{Path(name).stem}_{verdict}.jpg"),
                np.vstack([title, combo]))


def select_primary(boxes, h, w, alpha, beta):
    if len(boxes) == 0:
        return None, -1
    max_area = max((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
    best, bi, bs = None, -1, -1e9
    for i, b in enumerate(boxes):
        s = sp_score(b, h, w, max_area, alpha, beta)
        if s > bs:
            bs, best, bi = s, b, i
    return best, bi


def main():
    gt = json.load(open(GT_FILE, encoding="utf-8"))
    model = YOLO(str(YOLO_WEIGHTS))
    if hasattr(model.model, "to"):
        model.model.to(DEVICE)

    rows = []
    changed = []
    def_c = opt_c = n = 0

    for name in tqdm(gt.keys(), desc="YOLO"):
        p = INPUT_DIR / name
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model.predict(img, verbose=False, imgsz=960)
        boxes = res[0].boxes.xyxy.cpu().numpy() if (res and res[0].boxes is not None) else []
        if len(boxes) == 0:
            continue
        n += 1
        g = gt[name]

        pd, di = select_primary(boxes, h, w, *DEFAULT)
        po, oi = select_primary(boxes, h, w, *OPTIMAL)
        cd = iou(pd, g) > 0.5
        co = iou(po, g) > 0.5
        def_c += cd
        opt_c += co

        if SAVE_VIZ:                       # 每張都出圖:依 optimal 對/錯分 correct/ 或 wrong/
            save_single(img, boxes, oi, g, name, co)

        if di != oi:
            verdict = ("FIXED" if (not cd and co) else
                       "BROKE" if (cd and not co) else
                       "still_wrong" if not co else "still_right")
            changed.append((name, di, oi, cd, co, len(boxes), verdict))
            if SAVE_VIZ:
                save_compare(img, boxes, di, oi, g, name, verdict)

        rows.append([name, len(boxes), di, oi, int(cd), int(co)])

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["image", "n_boxes", "default_box", "optimal_box",
                      "default_correct", "optimal_correct"])
        wtr.writerows(rows)

    print(f"\nGT images processed: {n}")
    print(f"Default (a=1.0, b=1.5): {def_c}/{n} = {def_c/n*100:.2f}%")
    print(f"Optimal (a=2.0, b=1.5): {opt_c}/{n} = {opt_c/n*100:.2f}%")
    print(f"\nSelection changed in {len(changed)} image(s):")
    print(f"{'image':22}{'def':>5}{'opt':>5}{'defOK':>7}{'optOK':>7}{'nbox':>6}  verdict")
    for name, di, oi, cd, co, nb, v in changed:
        print(f"{name:22}{di:>5}{oi:>5}{str(bool(cd)):>7}{str(bool(co)):>7}{nb:>6}  {v}")
    print(f"\nPer-image CSV -> {OUT_CSV}")
    if SAVE_VIZ:
        print(f"Visualizations -> {VIZ_DIR}/")
        print(f"    correct/ : {opt_c} images   wrong/ : {n - opt_c} images"
              f"   changed/ : {len(changed)} images")


if __name__ == "__main__":
    main()
