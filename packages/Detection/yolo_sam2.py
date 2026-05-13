import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'
print(torch.cuda.device_count())
# ===========================
# 使用者需設定的參數
# ===========================
YOLO_WEIGHTS = "training_results/flower_detect/FD11/weights/best.pt"  # 你的 YOLO 權重（花朵偵測）
TEST_DIR     = "processing"                    # 測試圖片資料夾（裡面放多張 .jpg/.png）
IMG_SIZE     = 960                         # YOLO 推論解析度
CONF_THRES   = 0.5                       # YOLO 信心門檻
IOU_THRES    = 0.75                        # YOLO IoU 門檻
MAX_DET      = 100                       # YOLO 單張最多目標數
ALPHA_OVERLAY = 0.3                       # 總覽合成圖的透明度（0~1）

# 你給的 SAM2 載入方式與裝置：
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

def select_primary_flower(boxes_xyxy, img_h, img_w, alpha=1.0, beta=1.5):
    """
    Spatial Prominence-based Target Selection (SPTS).

    Ranks each detected bounding box by a prominence score:
        S_i = alpha * A_hat_i  -  beta * D_hat_i

    where:
        A_hat_i = box area / max(box areas)       (normalized area)
        D_hat_i = dist(box_center, img_center) / img_diagonal  (normalized distance)

    Returns the single box with the highest score, or None if input is empty.
    """
    if len(boxes_xyxy) == 0:
        return None

    img_cx, img_cy = img_w / 2.0, img_h / 2.0
    img_diag = np.sqrt(img_w ** 2 + img_h ** 2)

    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    max_area = areas.max()

    box_cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    box_cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    dists = np.sqrt((box_cx - img_cx) ** 2 + (box_cy - img_cy) ** 2)

    norm_area = areas / max_area
    norm_dist = dists / img_diag

    scores = alpha * norm_area - beta * norm_dist
    best_idx = np.argmax(scores)

    return boxes_xyxy[best_idx : best_idx + 1]  # (1, 4)

# ===== SAM2 Predictor（依你提供的 from_pretrained 方式）=====
class SAM2Predictor:
    """
    封裝 SAM2ImagePredictor 的簡單介面：
      - set_image(image_bgr)
      - predict_with_boxes(boxes_xyxy) -> list of masks (H,W) bool
    """
    def __init__(self, device: str = "cuda:3"):

        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        # 移動到指定裝置（多卡時很重要）
        if hasattr(self.predictor, "to"):
            self.predictor.to(device)
        self.device = device
        self.original_size = None

    def set_image(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # 多數 SAM/SAM2 預測器都支援 set_image
        self.predictor.set_image(image_rgb)
        self.original_size = image_rgb.shape[:2]  # (H, W)

    def predict_with_boxes(self, boxes_xyxy: np.ndarray):
        """
        boxes_xyxy: (N, 4) float32, in original image pixel coords [x1,y1,x2,y2]
        回傳 list[np.ndarray]，每個是 (H, W) 的 bool mask
        """
        masks_list = []
        # 逐一給 box，取單一候選（multimask_output=False）
        # 若你的 SAM2 需要 tensor，轉一下；否則維持 numpy 也可。
        for i in range(len(boxes_xyxy)):
            box = boxes_xyxy[i].astype(np.float32)
            # SAM2 大多延用 SAM v1 的 API 名稱；若你的版不同，改成對應的參數即可：
            # 例如某些版本寫法是：masks, scores, logits = predictor.predict(box=box[None, :], multimask_output=False)
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,                  # xyxy
                multimask_output=False    # 只要一張 mask
            )
            # 統一成 (H, W) 的 bool
            # 常見回傳 shape: (1, H, W)
            m = masks[0]
            if m.dtype != np.bool_:
                m = m.astype(bool)
            masks_list.append(m)
        return masks_list


# ===== 視覺化工具 =====
def random_color(seed=None):
    rng = np.random.default_rng(seed)
    return (int(rng.integers(0, 255)), int(rng.integers(0, 255)), int(rng.integers(0, 255)))

def apply_mask_color_keep_object(image_bgr, mask_bool):
    """
    只保留 mask 內的彩色內容，其他變黑
    """
    out = np.zeros_like(image_bgr)
    out[mask_bool] = image_bgr[mask_bool]
    return out

def overlay_instances(image_bgr, masks, alpha=0.45):
    """
    建立總覽合成圖：不同實例不同顏色半透明覆蓋
    """
    canvas = np.zeros_like(image_bgr, dtype=np.uint8)
    for i, m in enumerate(masks):
        c = random_color(seed=i)
        canvas[m] = c
    blended = cv2.addWeighted(image_bgr, 1 - alpha, canvas, alpha, 0)
    return blended


# ===== 主流程 =====
def main():
    # 1) YOLO
    model = YOLO(YOLO_WEIGHTS)

    # 2) SAM2
    sam2 = SAM2Predictor(device=device)
    print("SAM2 (facebook/sam2-hiera-large) loaded on", device)

    # 3) 蒐集圖片
    test_dir = Path(TEST_DIR)
    if not test_dir.exists():
        raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_paths = [p for p in test_dir.iterdir() if p.suffix.lower() in exts]
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {TEST_DIR}")

    print(f"[INFO] Found {len(img_paths)} images in {TEST_DIR}")

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue
        h, w, _ = img_bgr.shape 

        # 在原資料夾下建子資料夾
        out_dir = img_path.parent / f"{img_path.stem}_segments"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 4) YOLO 偵測
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMG_SIZE,
            max_det=MAX_DET,
            device=device if device.startswith("cuda:3") else 2,
            verbose=False
        )
        if not results:
            print(f"[INFO] No YOLO result for {img_path.name}")
            continue

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            print(f"[INFO] No detections for {img_path.name}")
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)  # (N,4)

        # ✅ Spatial Prominence-based Target Selection (SPTS)
        #    選出面積最大且最靠近畫面中心的唯一主花
        primary_box = select_primary_flower(boxes_xyxy, img_h=h, img_w=w)

        if primary_box is None:
            print(f"[INFO] No primary flower identified, skip {img_path.name}")
            continue

        filtered_boxes = primary_box
        
        # 5) SAM2：以 YOLO box 當 prompt 出 masks
        sam2.set_image(img_bgr)
        masks_list = sam2.predict_with_boxes(filtered_boxes)

        # 6) 輸出每朵花
        for i, mask in enumerate(masks_list, start=1):
            flower_img = apply_mask_color_keep_object(img_bgr, mask)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_flower_{i:03d}.png"), flower_img)

         # ✅ 6.5) 合併所有 segment 成一張前景圖
        if len(masks_list) > 0:
            merged_mask = np.zeros(img_bgr.shape[:2], dtype=bool)  # (H, W)
            for m in masks_list:
                merged_mask |= m.astype(bool)

            merged_img = apply_mask_color_keep_object(img_bgr, merged_mask)
            # 檔名你可以選：stem.png / stem_merge.png 自己決定
            cv2.imwrite(str(out_dir / f"{img_path.stem}.png"), merged_img)
            # 或 cv2.imwrite(str(out_dir / f"{img_path.stem}_merge.png"), merged_img)
        
        # 7) 輸出總覽合成圖
        overlay_img = overlay_instances(img_bgr, [m.astype(bool) for m in masks_list], alpha=ALPHA_OVERLAY)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_overlay.png"), overlay_img)

        print(f"[✓] {img_path.name}: {len(masks_list)} instances -> {out_dir}")

    print("[DONE] All images processed.")


if __name__ == "__main__":
    main()