"""
Shape Quantification - Trimmed Base Landmark Strategy
=====================================================
策略:
1. Mask 去噪: 只保留 target class 中面積最大的輪廓
2. Base = Column 質心, Tip = 輪廓上離 Base 最遠的點
3. Base->Tip 中軸線前 20% 處做法線, 捨棄法線以下的輪廓點
4. 剩餘輪廓等距重採樣到固定 N 個 landmarks

輸出: ./Datasets/test_dataset/viz/
  - viz_denoise_{stem}.png  : 去噪前後對比
  - viz_trim_{stem}.png     : 軸線/法線/trim/landmarks
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
VIZ_DIR = os.path.join(DATASET_DIR, 'viz')

TARGET_CLASS = 7   # Petal_R
BASE_CLASS = 1     # Column
TRIM_RATIO = 0.20  # 捨棄基部前 20%
NUM_LANDMARKS = 100  # 固定採樣數


# ==========================================
# 核心函式
# ==========================================

def get_column_centroid(mask):
    """取得 Column (class 1) 的質心座標"""
    mask_col = np.where(mask == BASE_CLASS, 255, 0).astype(np.uint8)
    M = cv2.moments(mask_col)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return np.array([cx, cy], dtype=float)


def denoise_mask(mask, target_class):
    """
    只保留 target_class 中面積最大的輪廓，其餘清除。
    回傳: (clean_binary_mask, all_contours, largest_contour)
    """
    binary = np.where(mask == target_class, 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, [], None

    largest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(binary)
    cv2.drawContours(clean, [largest], -1, 255, cv2.FILLED)

    return clean, cnts, largest


def find_tip(contour, base_pt):
    """輪廓上離 base_pt 最遠的點索引"""
    max_d, tip_i = 0, 0
    for i, pt in enumerate(contour):
        d = np.sum((pt[0].astype(float) - base_pt) ** 2)
        if d > max_d:
            max_d, tip_i = d, i
    return tip_i


def trim_contour(contour, base_pt, tip_pt, trim_ratio):
    """
    以 base_pt->tip_pt 為中軸, 在 trim_ratio 處做法線切割。
    回傳: (keep_pts, discard_pts, axis_unit, cut_point)
    """
    axis = tip_pt - base_pt
    axis_len = np.linalg.norm(axis)
    if axis_len == 0:
        return None, None, None, None

    axis_unit = axis / axis_len
    cut_dist = trim_ratio * axis_len
    cut_point = base_pt + cut_dist * axis_unit

    keep, discard = [], []
    for pt in contour:
        p = pt[0].astype(float)
        proj = np.dot(p - base_pt, axis_unit)
        if proj >= cut_dist:
            keep.append(p)
        else:
            discard.append(p)

    keep = np.array(keep) if keep else None
    discard = np.array(discard) if discard else None
    return keep, discard, axis_unit, cut_point


def resample_contour(points, n_pts):
    """
    將 2D 點序列等距重採樣到 n_pts 個點。
    使用弧長參數化 + spline 內插。
    """
    if points is None or len(points) < 4:
        return None

    # 計算累積弧長
    diffs = np.diff(points, axis=0)
    seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len == 0:
        return None

    # 正規化到 [0, 1]
    t = cum_len / total_len

    # spline 內插 (x 和 y 各自對 t)
    try:
        tck_x = interp.splrep(t, points[:, 0], s=0, k=3)
        tck_y = interp.splrep(t, points[:, 1], s=0, k=3)
        t_new = np.linspace(0, 1, n_pts)
        x_new = interp.splev(t_new, tck_x)
        y_new = interp.splev(t_new, tck_y)
        return np.column_stack([x_new, y_new])
    except Exception as e:
        print(f"  Resample failed: {e}")
        return None


# ==========================================
# Pipeline
# ==========================================

def process_one(img_bgr, mask, stem):
    """
    處理單張影像，回傳 landmarks 和視覺化資訊。
    """
    # 1) Column 質心
    base_pt = get_column_centroid(mask)
    if base_pt is None:
        return None, "no Column"

    # 2) 去噪: 只留最大花瓣
    clean_mask, all_cnts, largest_cnt = denoise_mask(mask, TARGET_CLASS)
    if largest_cnt is None:
        return None, "no Petal"

    # 3) 找 Tip
    tip_i = find_tip(largest_cnt, base_pt)
    tip_pt = largest_cnt[tip_i][0].astype(float)

    # 4) Trim
    keep, discard, axis_unit, cut_point = trim_contour(
        largest_cnt, base_pt, tip_pt, TRIM_RATIO)
    if keep is None or len(keep) < 10:
        return None, "too few points after trim"

    # 5) 等距重採樣到固定數量
    landmarks = resample_contour(keep, NUM_LANDMARKS)
    if landmarks is None:
        return None, "resample failed"

    info = {
        'base_pt': base_pt,
        'tip_pt': tip_pt,
        'axis_unit': axis_unit,
        'cut_point': cut_point,
        'all_cnts': all_cnts,
        'largest_cnt': largest_cnt,
        'clean_mask': clean_mask,
        'keep': keep,
        'discard': discard,
    }
    return landmarks, info


# ==========================================
# 視覺化
# ==========================================

def save_denoise_viz(img_bgr, mask, info, save_path, title):
    """去噪前後對比圖"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 左: 原始 mask 所有輪廓
    axes[0].imshow(img_rgb)
    raw_binary = np.where(mask == TARGET_CLASS, 255, 0).astype(np.uint8)
    raw_overlay = np.zeros_like(img_rgb)
    raw_overlay[:, :, 0] = raw_binary  # red channel
    axes[0].imshow(raw_overlay, alpha=0.4)
    for cnt in info['all_cnts']:
        axes[0].plot(cnt[:, 0, 0], cnt[:, 0, 1], 'r-', lw=1, alpha=0.8)
    axes[0].set_title(f'{title} - Raw Mask ({len(info["all_cnts"])} contours)', fontsize=12)
    axes[0].axis('off')

    # 中: 去噪後 (只留最大)
    axes[1].imshow(img_rgb)
    clean_overlay = np.zeros_like(img_rgb)
    clean_overlay[:, :, 1] = info['clean_mask']  # green channel
    axes[1].imshow(clean_overlay, alpha=0.4)
    axes[1].plot(info['largest_cnt'][:, 0, 0], info['largest_cnt'][:, 0, 1],
                 'g-', lw=2)
    axes[1].set_title(f'{title} - Denoised (largest only)', fontsize=12)
    axes[1].axis('off')

    # 右: Column 質心標示
    axes[2].imshow(img_rgb)
    clean_overlay2 = np.zeros_like(img_rgb)
    clean_overlay2[:, :, 1] = info['clean_mask']
    axes[2].imshow(clean_overlay2, alpha=0.3)
    axes[2].plot(*info['base_pt'], 'r*', ms=15, label='Column Centroid (Base)')
    axes[2].plot(*info['tip_pt'], 'b^', ms=12, label='Tip')
    axes[2].legend(fontsize=9, loc='upper right')
    axes[2].set_title(f'{title} - Base & Tip', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('Step 1: Mask Denoise & Base/Tip', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_trim_viz(img_bgr, landmarks, info, save_path, title):
    """Trim + 重採樣結果圖"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    base = info['base_pt']
    tip = info['tip_pt']
    au = info['axis_unit']
    cut = info['cut_point']

    # 左: 中軸 + 法線
    axes[0].imshow(img_rgb)
    cnt = info['largest_cnt']
    axes[0].plot(cnt[:, 0, 0], cnt[:, 0, 1], 'c-', lw=1.5, label='Contour')
    axes[0].plot(*base, 'r*', ms=15, label='Base (Column Centroid)')
    axes[0].plot(*tip, 'b^', ms=12, label='Tip')
    axes[0].plot([base[0], tip[0]], [base[1], tip[1]], 'g--', lw=2, label='Axis')
    perp = np.array([-au[1], au[0]])
    L = 250
    axes[0].plot([cut[0]-L*perp[0], cut[0]+L*perp[0]],
                 [cut[1]-L*perp[1], cut[1]+L*perp[1]],
                 'm-', lw=2.5, label=f'{int(TRIM_RATIO*100)}% Cut')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_title(f'{title} - Axis & Cut Line', fontsize=12)
    axes[0].axis('off')

    # 中: Keep vs Discard
    axes[1].imshow(img_rgb)
    if info['discard'] is not None:
        axes[1].scatter(info['discard'][:, 0], info['discard'][:, 1],
                        c='red', s=6, alpha=0.7, label='Discarded')
    axes[1].scatter(info['keep'][:, 0], info['keep'][:, 1],
                    c='lime', s=6, alpha=0.9, label='Kept')
    axes[1].set_title(f'{title} - Trim Result', fontsize=12)
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].axis('off')

    # 右: 重採樣後的固定 landmarks
    axes[2].imshow(img_rgb)
    axes[2].scatter(landmarks[:, 0], landmarks[:, 1], c='yellow', s=12, zorder=5)
    axes[2].plot(np.append(landmarks[:, 0], landmarks[0, 0]),
                 np.append(landmarks[:, 1], landmarks[0, 1]),
                 'y-', lw=1, alpha=0.6)
    axes[2].set_title(f'{title} - {len(landmarks)} Resampled Landmarks', fontsize=12)
    axes[2].axis('off')

    plt.suptitle(f'Step 2-3: Trim {int(TRIM_RATIO*100)}% & Resample to {NUM_LANDMARKS}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ==========================================
# 主程式
# ==========================================
if __name__ == '__main__':
    os.makedirs(VIZ_DIR, exist_ok=True)

    img_files = sorted([f for f in os.listdir(IMAGE_DIR)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"[*] Found {len(img_files)} images.")

    ok, fail = 0, 0

    for img_file in img_files:
        stem = os.path.splitext(img_file)[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, img_file))
        mask = cv2.imread(os.path.join(MASK_DIR, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"  [SKIP] {img_file}: file not found")
            fail += 1
            continue

        landmarks, result = process_one(img, mask, stem)

        if landmarks is None:
            print(f"  [SKIP] {img_file}: {result}")
            fail += 1
            continue

        # 存兩張視覺化
        save_denoise_viz(img, mask, result,
                         os.path.join(VIZ_DIR, f"viz_denoise_{stem}.png"), stem)
        save_trim_viz(img, landmarks, result,
                      os.path.join(VIZ_DIR, f"viz_trim_{stem}.png"), stem)

        ok += 1
        print(f"  [OK] {stem}: {len(landmarks)} landmarks")

    print(f"\n{'='*60}")
    print(f"[Done] OK={ok}  FAIL={fail}")
    print(f"[Fixed landmark count] = {NUM_LANDMARKS}")
    print(f"[Output] {VIZ_DIR}")
