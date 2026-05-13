import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def visualize_spts(img_w=800, img_h=600, alpha=1.0, beta=1.5):
    """
    視覺化 SPTS (Spatial Prominence-based Target Selection) 演算法。
    """
    img_cx, img_cy = img_w / 2.0, img_h / 2.0
    img_diag = np.sqrt(img_w ** 2 + img_h ** 2)

    # 模擬 5 個 YOLO 偵測到的 bounding boxes [x1, y1, x2, y2]
    boxes = np.array([
        [50, 50, 150, 150],       # 小花，左上角
        [600, 400, 780, 580],     # 中型花，右下角
        [200, 150, 600, 450],     # 大花，接近中心 (預期會被選為 Primary Flower)
        [350, 50, 450, 150],      # 小花，正上方
        [50, 300, 250, 500]       # 中型花，左方
    ], dtype=np.float32)

    # 計算各項指標
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    max_area = areas.max()

    box_cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    box_cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    dists = np.sqrt((box_cx - img_cx) ** 2 + (box_cy - img_cy) ** 2)

    norm_area = areas / max_area
    norm_dist = dists / img_diag

    scores = alpha * norm_area - beta * norm_dist
    best_idx = np.argmax(scores)

    # 計算排名並印出文字結果
    sorted_indices = np.argsort(scores)[::-1]
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(scores) + 1)

    print("\n--- SPTS Box Evaluation Results ---")
    for i in sorted_indices:
        print(f"Rank {ranks[i]}: Box {i} | Score: {scores[i]:.3f} (Area: {norm_area[i]:.3f}, Dist: {norm_dist[i]:.3f})")
    print("-----------------------------------\n")

    # 開始繪圖
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0) # 反轉 Y 軸符合影像座標
    ax.set_facecolor('#f0f0f0') # 淺灰背景

    # 標示影像中心點
    ax.plot(img_cx, img_cy, 'r+', markersize=20, markeredgewidth=2, label='Image Center')

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        w = x2 - x1
        h = y2 - y1
        
        is_best = (i == best_idx)
        color = 'red' if is_best else 'gray'
        linewidth = 4 if is_best else 2
        linestyle = '-' if is_best else '--'
        alpha_fill = 0.2 if is_best else 0.05
        
        # 繪製 Bounding Box
        rect = patches.Rectangle((x1, y1), w, h, linewidth=linewidth, edgecolor=color, facecolor=color, alpha=alpha_fill, linestyle=linestyle)
        ax.add_patch(rect)
        rect_outline = patches.Rectangle((x1, y1), w, h, linewidth=linewidth, edgecolor=color, facecolor='none', linestyle=linestyle)
        ax.add_patch(rect_outline)
        
        # 標示 Box 中心點
        ax.plot(box_cx[i], box_cy[i], marker='o', color=color, markersize=6)
        
        # 連接 Box 中心到影像中心的距離線
        ax.plot([img_cx, box_cx[i]], [img_cy, box_cy[i]], color=color, linestyle=':', alpha=0.6)
        
        # 顯示數值
        label = (f"Rank: {ranks[i]}\n"
                 f"Score: {scores[i]:.2f}\n"
                 f"Area: {norm_area[i]:.2f}\n"
                 f"Dist: {norm_dist[i]:.2f}")
        
        # 最佳的主花加上特殊標籤
        if is_best:
            label = "★ PRIMARY FLOWER ★\n" + label
            
        ax.text(x1, y1-5, label, color=color, fontsize=10, fontweight='bold' if is_best else 'normal', verticalalignment='bottom')

    ax.set_title(f"SPTS Algorithm Visualization (alpha={alpha}, beta={beta})", fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    
    # 輸出儲存
    output_path = 'SPTS_Visualization.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"SPTS Visualization output path: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    visualize_spts()
