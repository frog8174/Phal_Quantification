import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import os

# ==========================================
# 設定與路徑
# ==========================================
DATASET_DIR = './Datasets/test_dataset'
IMAGE_DIR = os.path.join(DATASET_DIR, 'images')
MASK_DIR = os.path.join(DATASET_DIR, 'masks')
VIZ_DIR = os.path.join(DATASET_DIR, 'viz_4panel')

TARGET_CLASS_ID = 7   # 右花瓣
BASE_CLASS_ID = 1     # 蕊柱
LABELLUM_CLASS_ID = 3 # 唇瓣 (用來偵測遮擋)
NUM_LANDMARKS = 100
PADDING = 50

# ==========================================
# 核心函式 (與 notebook 一致)
# ==========================================
def get_column_centroid(mask):
    mask_col = np.where(mask == BASE_CLASS_ID, 255, 0).astype(np.uint8)
    M = cv2.moments(mask_col)
    if M['m00'] == 0: return None
    return np.array([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])

def fix_occlusion(contour, mask):
    labellum = np.where(mask == LABELLUM_CLASS_ID, 255, 0).astype(np.uint8)
    if cv2.countNonZero(labellum) == 0: return contour, None
    labellum_dilated = cv2.dilate(labellum, np.ones((11, 11), np.uint8), iterations=1)
    
    touching_idx = []
    for i in range(len(contour)):
        x, y = contour[i][0]
        if 0 <= y < labellum_dilated.shape[0] and 0 <= x < labellum_dilated.shape[1]:
            if labellum_dilated[y, x] > 0: touching_idx.append(i)
            
    if len(touching_idx) < 5: return contour, None
    
    touching_idx = np.array(touching_idx)
    diffs = np.diff(touching_idx)
    wrap_gap = len(contour) - touching_idx[-1] + touching_idx[0]
    all_gaps = np.append(diffs, wrap_gap)
    max_gap_idx = np.argmax(all_gaps)
    
    if max_gap_idx == len(diffs):
        occ_start, occ_end = touching_idx[0], touching_idx[-1]
    else:
        occ_start, occ_end = touching_idx[max_gap_idx + 1], touching_idx[max_gap_idx]
        
    pt_start, pt_end = contour[occ_start][0], contour[occ_end][0]
    
    if occ_start <= occ_end:
        valid_contour = np.concatenate([contour[occ_end:], contour[:occ_start+1]])
    else:
        valid_contour = contour[occ_end:occ_start+1]
        
    num_interp = 20
    xs, ys = np.linspace(pt_start[0], pt_end[0], num_interp), np.linspace(pt_start[1], pt_end[1], num_interp)
    patch_pts = np.column_stack((xs, ys)).reshape(-1, 1, 2).astype(np.int32)
    patch_pts = patch_pts[1:-1] if num_interp > 2 else np.empty((0, 1, 2), dtype=np.int32)
    return np.concatenate([valid_contour, patch_pts]), {'occ_start': pt_start, 'occ_end': pt_end}

def resample_path(points, n_pts):
    if len(points) < 2: return None
    
    # 計算相鄰點的歐式距離
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    
    # 計算累積弧長 (Cumulative Arc Length)
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dist[-1]
    
    if total_dist == 0:
        return np.tile(points[0], (n_pts, 1))
        
    # 等距採樣目標距離
    target_dist = np.linspace(0, total_dist, n_pts)
    
    # 對 x, y 分別做一維線性內插
    new_x = np.interp(target_dist, cum_dist, points[:, 0])
    new_y = np.interp(target_dist, cum_dist, points[:, 1])
    
    return np.column_stack([new_x, new_y])

# ==========================================
# 主程式
# ==========================================
if __name__ == '__main__':
    os.makedirs(VIZ_DIR, exist_ok=True)
    image_paths = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg'))])
    print(f'[*] 系統共抓取到 {len(image_paths)} 組檔案，準備輸出 4-panel 診斷圖...')

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        mask_path = os.path.join(MASK_DIR, f'{stem}.png')
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: continue

        # 轉換顏色空間並補邊緣
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_padded = cv2.copyMakeBorder(img, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        mask_padded = cv2.copyMakeBorder(mask, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=0)
        
        col_centroid = get_column_centroid(mask_padded)
        if col_centroid is None: continue

        mask_c7 = np.where(mask_padded == TARGET_CLASS_ID, 255, 0).astype(np.uint8)
        mask_c7 = cv2.morphologyEx(mask_c7, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts_c7, _ = cv2.findContours(mask_c7, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_c7: continue

        cnt = max(cnts_c7, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 100 or len(cnt) < 20: continue

        patched_cnt, occ_info = fix_occlusion(cnt, mask_padded)
            
        min_dist, base_idx = float('inf'), 0
        for idx, pt in enumerate(patched_cnt):
            dist = np.sum((pt[0].astype(float) - col_centroid)**2)
            if dist < min_dist: min_dist, base_idx = dist, idx
                
        base_pt = patched_cnt[base_idx][0].astype(float)
        max_dist, tip_idx = 0, 0
        for idx, pt in enumerate(patched_cnt):
            dist = np.sum((pt[0].astype(float) - base_pt)**2)
            if dist > max_dist: max_dist, tip_idx = dist, idx

        if base_idx == tip_idx: continue

        if base_idx < tip_idx:
            path1 = patched_cnt[base_idx : tip_idx+1][:, 0, :]
            path2 = np.concatenate([patched_cnt[tip_idx:], patched_cnt[:base_idx+1]])[:, 0, :]
        else:
            path1 = np.concatenate([patched_cnt[base_idx:], patched_cnt[:tip_idx+1]])[:, 0, :]
            path2 = patched_cnt[tip_idx : base_idx+1][:, 0, :]

        n_side = (NUM_LANDMARKS // 2) + 1
        lm1, lm2 = resample_path(path1, n_side), resample_path(path2, n_side)
        if lm1 is None or lm2 is None: continue
            
        final_landmarks = np.concatenate([lm1[:-1], lm2[:-1]])
        
        # ==========================================
        # 繪圖 (4-Panel)
        # ==========================================
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f"Sample: {filename}", fontsize=18, fontweight='bold')
        
        # [Panel 1] Original
        axes[0].imshow(img_padded)
        axes[0].set_title("1. Original Image", fontsize=14)
        axes[0].axis('off')
        
        # [Panel 2] Mask
        colored_mask = np.zeros_like(img_padded)
        colored_mask[mask_padded == TARGET_CLASS_ID] = [255, 105, 180] # 粉紅: 花瓣
        colored_mask[mask_padded == BASE_CLASS_ID] = [255, 255, 0]     # 黃: 蕊柱
        colored_mask[mask_padded == LABELLUM_CLASS_ID] = [0, 191, 255] # 藍: 唇瓣
        
        blended = cv2.addWeighted(img_padded, 0.4, colored_mask, 0.6, 0)
        axes[1].imshow(blended)
        axes[1].set_title("2. Segmentation Mask", fontsize=14)
        axes[1].axis('off')
        
        # [Panel 3] Contour & Occlusion
        axes[2].imshow(img_padded)
        axes[2].plot(cnt[:, 0, 0], cnt[:, 0, 1], 'c-', lw=2, label='Raw Contour')
        if occ_info:
            p_start, p_end = occ_info['occ_start'], occ_info['occ_end']
            axes[2].plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', lw=2.5, label='Patched Line')
            axes[2].plot(*p_start, 'ro', ms=6)
            axes[2].plot(*p_end, 'rs', ms=6)
            
        axes[2].plot(*col_centroid, 'y*', ms=12, label='Column Centroid')
        axes[2].plot(*base_pt, 'ro', ms=10, label='Base')
        axes[2].plot(*patched_cnt[tip_idx][0], 'b^', ms=10, label='Tip')
        
        axes[2].set_title("3. Contour & Occlusion Patch", fontsize=14)
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].axis('off')
        
        # [Panel 4] Landmarks
        axes[3].imshow(img_padded)
        closed_x = np.append(final_landmarks[:, 0], final_landmarks[0, 0])
        closed_y = np.append(final_landmarks[:, 1], final_landmarks[0, 1])
        axes[3].plot(closed_x, closed_y, 'g-', lw=1.5, alpha=0.7)
        axes[3].scatter(final_landmarks[:, 0], final_landmarks[:, 1], c='lime', s=15, edgecolors='black', linewidth=0.5, zorder=5)
        
        axes[3].plot(final_landmarks[0, 0], final_landmarks[0, 1], 'ro', ms=12, label='0 (Base)', zorder=6)
        axes[3].plot(final_landmarks[50, 0], final_landmarks[50, 1], 'b^', ms=12, label='50 (Tip)', zorder=6)
        
        axes[3].set_title("4. Spline Landmarks (100 pts)", fontsize=14)
        axes[3].legend(loc='upper right', fontsize=10)
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(VIZ_DIR, f"4panel_{stem}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  [Saved] {save_path}")
        
    print(f"\n[Done] 全部完成，圖檔儲存於: {VIZ_DIR}")
