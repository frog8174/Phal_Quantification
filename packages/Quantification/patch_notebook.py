"""
Programmatically modify ColorQuantify.ipynb:
1. Cell 03: Add Kneedle algorithm for objective K selection
2. Cell 04: Change FINAL_K = 8 -> 5 (or auto from Kneedle)
3. Add new Cell: Donut Chart generation
"""

import json
import copy

NOTEBOOK_PATH = r'y:\Workspace\Aaron\2026-CIGR-phal-yolo-seg-quantify\packages\Quantification\ColorQuantify.ipynb'

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ══════════════════════════════════════════════════════════
# Cell 03 — K Evaluation: 加入 Kneedle 演算法
# ══════════════════════════════════════════════════════════
cell03_new_source = r'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import os

output_dir = './Pigment'

# 1. 讀取 Cell 1 存下來的資料
print("[*] [Cell 2] 讀取平衡後的特徵資料...")
balanced_lab = np.load(os.path.join(output_dir, "balanced_lab.npy"))

# 2. 測試 K=2 到 K=15
k_range = list(range(2, 16))
sse, sil_scores = [], []

print(f"[*] 開始計算 SSE 與 Silhouette Score ... (這可能需要幾分鐘)")
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans_test.fit_predict(balanced_lab)
    
    sse.append(kmeans_test.inertia_)
    score = silhouette_score(balanced_lab, labels, sample_size=5000, random_state=42)
    sil_scores.append(score)
    print(f"  ▶ K={k:2d} | SSE: {kmeans_test.inertia_:,.0f} | Silhouette: {score:.4f}")

# 3. Kneedle 演算法自動偵測 Elbow Point
kn = KneeLocator(k_range, sse, curve='convex', direction='decreasing')
optimal_k_kneedle = kn.knee
optimal_k_sil = k_range[np.argmax(sil_scores)]

print(f"\n[💡] Kneedle 偵測 Elbow Point: K = {optimal_k_kneedle}")
print(f"[💡] Silhouette 最高峰: K = {optimal_k_sil}")

# 印出 Kneedle line distance (供論文引用)
if hasattr(kn, 'all_knees_y') and hasattr(kn, 'all_norm_distances'):
    for k_val, dist in zip(k_range, kn.y_difference):
        print(f"  K={k_val:2d} | Kneedle line distance: {dist:.4f}")

# 4. 繪製三合一圖表
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

# SSE (左軸)
ax1.set_xlabel('Number of Clusters (K)', fontsize=14)
ax1.set_ylabel('Sum of Squared Errors (SSE)', color='tab:blue', fontsize=14)
ax1.plot(k_range, sse, marker='o', color='tab:blue', linewidth=2, label='SSE')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(k_range)
ax1.grid(True, linestyle='--', alpha=0.5)

# Silhouette (右軸)
ax2 = ax1.twinx()  
ax2.set_ylabel('Silhouette Score', color='tab:red', fontsize=14)
ax2.plot(k_range, sil_scores, marker='s', color='tab:red', linewidth=2, label='Silhouette')
ax2.tick_params(axis='y', labelcolor='tab:red')

# 標記 Kneedle K_opt
ax1.axvline(x=optimal_k_kneedle, color='green', linestyle='--', alpha=0.8, linewidth=2)
ax1.text(optimal_k_kneedle + 0.3, max(sse)*0.85, f'Kneedle\nK={optimal_k_kneedle}', 
         color='green', fontsize=12, fontweight='bold')

# 標記 Silhouette max
ax2.axvline(x=optimal_k_sil, color='red', linestyle=':', alpha=0.6)
ax2.text(optimal_k_sil + 0.3, max(sil_scores)*0.95, f'Max Silhouette\nK={optimal_k_sil}', 
         color='red', fontsize=10)

plt.title("Objective Evaluation of Optimal K\n(Kneedle Algorithm + Silhouette Score)", 
          fontsize=16, fontweight='bold', pad=15)
plt.savefig(os.path.join(output_dir, "02_K_Evaluation.png"), bbox_inches='tight')
plt.show()

print(f"\n[✔] 建議使用 K = {optimal_k_kneedle} (Kneedle 演算法客觀偵測)")
'''

nb['cells'][3]['source'] = cell03_new_source.split('\n')
nb['cells'][3]['source'] = [line + '\n' for line in nb['cells'][3]['source'][:-1]] + [nb['cells'][3]['source'][-1]]

# ══════════════════════════════════════════════════════════
# Cell 04 — Palette Generation: K=8 → K=5
# ══════════════════════════════════════════════════════════
cell04_new_source = r'''import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import os

output_dir = './Pigment'
balanced_lab = np.load(os.path.join(output_dir, "balanced_lab.npy"))

# 🌟 根據 Kneedle 演算法的客觀偵測結果，設定最終的 K 值
FINAL_K = 5

print(f"[*] [Cell 3] 正在使用 K={FINAL_K} 建立全域標準色盤...")
kmeans = KMeans(n_clusters=FINAL_K, init='k-means++', n_init=10, random_state=42)
kmeans.fit(balanced_lab)
global_centers_lab = kmeans.cluster_centers_

global_centers_rgb = []
for center in global_centers_lab:
    rgb_pixel = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_LAB2RGB)[0][0]
    global_centers_rgb.append(rgb_pixel)
global_centers_rgb = np.array(global_centers_rgb)

# 將標準尺存檔
np.save(os.path.join(output_dir, "global_centers_lab.npy"), global_centers_lab)
np.save(os.path.join(output_dir, "global_centers_rgb.npy"), global_centers_rgb)

# 畫出最終色盤
fig, ax = plt.subplots(figsize=(10, 2), dpi=150)
for i, rgb in enumerate(global_centers_rgb):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb/255.0))
    text_color = 'white' if (rgb[0]*0.299 + rgb[1]*0.587 + rgb[2]*0.114) < 128 else 'black'
    ax.text(i+0.5, 0.5, f"Bin_{i}", ha='center', va='center', color=text_color, fontweight='bold')

ax.set_xlim(0, FINAL_K)
ax.set_ylim(0, 1)
ax.axis('off')
plt.title(f"Global Standard Palette (K={FINAL_K})", fontsize=14, fontweight='bold')
plt.savefig(os.path.join(output_dir, "03_global_palette.png"))
plt.show()

print(f"[✔] 標準色盤已鎖定並存入 {output_dir}/")
'''

nb['cells'][4]['source'] = cell04_new_source.split('\n')
nb['cells'][4]['source'] = [line + '\n' for line in nb['cells'][4]['source'][:-1]] + [nb['cells'][4]['source'][-1]]

# ══════════════════════════════════════════════════════════
# New Cell — Donut Chart (插入在 Cell 05 之後)
# ══════════════════════════════════════════════════════════
donut_cell_source = r'''# Phase 6: Donut Chart Visualization
%run plot_color_donut.py
'''

donut_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + '\n' for line in donut_cell_source.split('\n')[:-1]] + [donut_cell_source.split('\n')[-1]]
}

# Insert after Cell 05 (index 5), before Cell 06 (empty cell)
nb['cells'].insert(6, donut_cell)

# ══════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done! ColorQuantify.ipynb has been updated:")
print("  - Cell 03: Added Kneedle algorithm (kneed.KneeLocator)")
print("  - Cell 04: FINAL_K changed from 8 to 5")
print("  - New Cell (after Cell 05): Donut chart via %run plot_color_donut.py")
