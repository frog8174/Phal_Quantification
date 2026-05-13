"""
Shape Quantification - Occlusion-Aware Landmark Strategy
=====================================================
策略:
1. 確認 Class 7 (Petal) 是否受 Class 3 (Labellum) 遮擋。
2. 若有遮擋，找出第一遮擋點與最後遮擋點，用直線填補該段受遮擋的輪廓。
3. Base = 輪廓上離 Column 質心最近的點，Tip = 輪廓上離 Base 最遠的點。
4. 以 Base 和 Tip 將輪廓分為左右兩半，各自內插相同數量的點 (N/2)。
5. 縫合為最終的 Landmarks (保證同源對應)。
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate as interp
import os

# ==========================================
# 設定
# ==========================================
DATASET_DIR = './Datasets/test_dataset'
IMAGE_DIR = os.path.join(DATASET_DIR, 'images')
MASK_DIR = os.path.join(DATASET_DIR, 'masks')
VIZ_DIR = os.path.join(DATASET_DIR, 'viz_occlusion')

TARGET_CLASS = 7   # Petal_R
BASE_CLASS = 1     # Column
LABELLUM_CLASS = 3 # Labellum (可能造成遮擋)
NUM_LANDMARKS = 100  # 固定採樣數
PADDING = 50       # 邊界防呆擴充 (像素)

# ==========================================
# 核心函式
# ==========================================

def get_column_centroid(mask):
    mask_col = np.where(mask == BASE_CLASS, 255, 0).astype(np.uint8)
    M = cv2.moments(mask_col)
    if M["m00"] == 0:
        return None
    return np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])


def denoise_mask(mask, target_class):
    binary = np.where(mask == target_class, 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def fix_occlusion(contour, mask, labellum_class=3):
    """
    偵測花瓣輪廓是否碰到 Labellum，若有，則用直線修補遮擋區域。
    """
    labellum = np.where(mask == labellum_class, 255, 0).astype(np.uint8)
    if cv2.countNonZero(labellum) == 0:
        return contour, None
        
    kernel = np.ones((11, 11), np.uint8)
    labellum_dilated = cv2.dilate(labellum, kernel, iterations=1)
    
    touching_idx = []
    for i in range(len(contour)):
        x, y = contour[i][0]
        # 防呆: 確保座標在影像範圍內
        if 0 <= y < labellum_dilated.shape[0] and 0 <= x < labellum_dilated.shape[1]:
            if labellum_dilated[y, x] > 0:
                touching_idx.append(i)
                
    if len(touching_idx) < 5:
        return contour, None # 碰觸點太少當作無遮擋
        
    touching_idx = np.array(touching_idx)
    diffs = np.diff(touching_idx)
    wrap_gap = len(contour) - touching_idx[-1] + touching_idx[0]
    
    all_gaps = np.append(diffs, wrap_gap)
    max_gap_idx = np.argmax(all_gaps)
    
    if max_gap_idx == len(diffs):
        # 遮擋連續出現在陣列中間
        occ_start = touching_idx[0]
        occ_end = touching_idx[-1]
    else:
        # 遮擋跨越了陣列的頭尾 (wrap-around)
        occ_start = touching_idx[max_gap_idx + 1]
        occ_end = touching_idx[max_gap_idx]
        
    pt_start = contour[occ_start][0]
    pt_end = contour[occ_end][0]
    
    # 擷取未被遮擋的有效輪廓 (從 occ_end 走到 occ_start)
    if occ_start <= occ_end:
        valid_contour = np.concatenate([contour[occ_end:], contour[:occ_start+1]])
    else:
        valid_contour = contour[occ_end:occ_start+1]
        
    # 用直線填補缺口
    num_interp = 20
    xs = np.linspace(pt_start[0], pt_end[0], num_interp)
    ys = np.linspace(pt_start[1], pt_end[1], num_interp)
    patch_pts = np.column_stack((xs, ys)).reshape(-1, 1, 2).astype(np.int32)
    
    if num_interp > 2:
        patch_pts = patch_pts[1:-1] # 移除頭尾避免與 valid_contour 重複
    else:
        patch_pts = np.empty((0, 1, 2), dtype=np.int32)
        
    patched_contour = np.concatenate([valid_contour, patch_pts])
    
    vis_info = {
        'occ_start': pt_start,
        'occ_end': pt_end,
    }
    
    return patched_contour, vis_info


def find_base_tip_indices(contour, column_centroid):
    min_dist = float('inf')
    base_idx = 0
    for i, pt in enumerate(contour):
        dist = np.sum((pt[0].astype(float) - column_centroid)**2)
        if dist < min_dist:
            min_dist = dist
            base_idx = i
            
    base_pt = contour[base_idx][0].astype(float)
    max_dist = 0
    tip_idx = 0
    for i, pt in enumerate(contour):
        dist = np.sum((pt[0].astype(float) - base_pt)**2)
        if dist > max_dist:
            max_dist = dist
            tip_idx = i
            
    return base_idx, tip_idx


def split_contour(contour, base_i, tip_i):
    if base_i < tip_i:
        path1 = contour[base_i:tip_i+1]
        path2 = np.concatenate([contour[tip_i:], contour[:base_i+1]])
    else:
        path1 = np.concatenate([contour[base_i:], contour[:tip_i+1]])
        path2 = contour[tip_i:base_i+1]
    
    return path1[:, 0, :], path2[:, 0, :]


def resample_path(points, n_pts):
    if points is None or len(points) < 2:
        return None
    
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
# Pipeline
# ==========================================

def process_one(img_bgr, mask):
    # 1. 找質心
    col_centroid = get_column_centroid(mask)
    if col_centroid is None: return None, "no Column"
        
    # 2. 取最大花瓣輪廓
    largest_cnt = denoise_mask(mask, TARGET_CLASS)
    if largest_cnt is None: return None, "no Petal"
        
    # 3. 修補遮擋
    patched_cnt, occ_info = fix_occlusion(largest_cnt, mask, LABELLUM_CLASS)
    
    # 4. 找 Base/Tip
    base_i, tip_i = find_base_tip_indices(patched_cnt, col_centroid)
    if base_i == tip_i: return None, "base and tip identical"
        
    # 5. 分左右半
    path1, path2 = split_contour(patched_cnt, base_i, tip_i)
    
    # 6. 等數量內插
    n_side = (NUM_LANDMARKS // 2) + 1
    lm1 = resample_path(path1, n_side) # idx: 0(Base) -> 50(Tip)
    lm2 = resample_path(path2, n_side) # idx: 0(Tip) -> 50(Base)
    
    if lm1 is None or lm2 is None: return None, "resample failed"
        
    # 縫合: lm1 去尾 (不含Tip), lm2 去尾 (不含Base)
    landmarks = np.concatenate([lm1[:-1], lm2[:-1]])
    
    info = {
        'col_centroid': col_centroid,
        'largest_cnt': largest_cnt,
        'patched_cnt': patched_cnt,
        'occ_info': occ_info,
        'path1': path1,
        'path2': path2,
        'base_pt': patched_cnt[base_i][0],
        'tip_pt': patched_cnt[tip_i][0]
    }
    
    return landmarks, info

# ==========================================
# 視覺化
# ==========================================

def save_viz(img_bgr, mask, landmarks, info, save_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Panel 1: Occlusion
    axes[0].imshow(img_rgb)
    lab_mask = np.where(mask == LABELLUM_CLASS, 255, 0).astype(np.uint8)
    overlay = np.zeros_like(img_rgb)
    overlay[:, :, 2] = lab_mask # Blue channel for Labellum
    axes[0].imshow(overlay, alpha=0.5)
    
    cnt = info['largest_cnt']
    axes[0].plot(cnt[:, 0, 0], cnt[:, 0, 1], 'c-', lw=1.5, label='Original Petal')
    
    if info['occ_info']:
        p_start = info['occ_info']['occ_start']
        p_end = info['occ_info']['occ_end']
        axes[0].plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', lw=2.5, label='Patched Line')
        axes[0].plot(*p_start, 'ro', ms=8)
        axes[0].plot(*p_end, 'rs', ms=8)
        axes[0].set_title(f"Occlusion Detection", fontsize=12)
    else:
        axes[0].set_title(f"No Occlusion Detected", fontsize=12)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].axis('off')
    
    # Panel 2: Base/Tip & Split Paths
    axes[1].imshow(img_rgb)
    axes[1].plot(info['path1'][:, 0], info['path1'][:, 1], 'm-', lw=2.5, label='Path 1 (Base->Tip)')
    axes[1].plot(info['path2'][:, 0], info['path2'][:, 1], 'g-', lw=2.5, label='Path 2 (Tip->Base)')
    
    axes[1].plot(*info['col_centroid'], 'y*', ms=15, label='Column Centroid')
    axes[1].plot(*info['base_pt'], 'ro', ms=10, label='Base')
    axes[1].plot(*info['tip_pt'], 'b^', ms=10, label='Tip')
    
    axes[1].set_title("Anchored Split", fontsize=12)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].axis('off')
    
    # Panel 3: Final Landmarks
    axes[2].imshow(img_rgb)
    axes[2].plot(np.append(landmarks[:, 0], landmarks[0, 0]), 
                 np.append(landmarks[:, 1], landmarks[0, 1]), 'y-', lw=1.5, alpha=0.6)
    axes[2].scatter(landmarks[:, 0], landmarks[:, 1], c='yellow', s=15, zorder=5)
    
    axes[2].plot(landmarks[0, 0], landmarks[0, 1], 'ro', ms=12, label='0 (Base)')
    tip_idx = NUM_LANDMARKS // 2
    axes[2].plot(landmarks[tip_idx, 0], landmarks[tip_idx, 1], 'b^', ms=12, label=f'{tip_idx} (Tip)')
    
    axes[2].set_title(f"Final {NUM_LANDMARKS} Landmarks", fontsize=12)
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].axis('off')
    
    plt.suptitle(f"{title} - Occlusion-Aware Landmark Strategy", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    os.makedirs(VIZ_DIR, exist_ok=True)
    img_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg'))])
    print(f"[*] Found {len(img_files)} images.")

    ok, fail = 0, 0
    for img_file in img_files:
        stem = os.path.splitext(img_file)[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, img_file))
        mask = cv2.imread(os.path.join(MASK_DIR, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        # 稍微 Padding 原圖與 mask 避免輪廓緊貼影像邊界造成判定異常
        img = cv2.copyMakeBorder(img, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        mask = cv2.copyMakeBorder(mask, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=0)

        landmarks, info = process_one(img, mask)
        if landmarks is None:
            print(f"  [FAIL] {stem}: {info}")
            fail += 1
            continue
            
        save_viz(img, mask, landmarks, info, os.path.join(VIZ_DIR, f"viz_occ_{stem}.png"), stem)
        print(f"  [OK] {stem}: {len(landmarks)} landmarks -> {info['occ_info'] is not None}")
        ok += 1

    print(f"\n[Done] OK={ok}  FAIL={fail}  OutputDir={VIZ_DIR}")
