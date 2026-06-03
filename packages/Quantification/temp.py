import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

base_class_id = 1
target_class_id = 7

mask_path = "packages/Quantification/Datasets/test_dataset/masks/001.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.copyMakeBorder(mask, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)

mask_c1 = np.where(mask == base_class_id, 255, 0).astype(np.uint8)
M_c1 = cv2.moments(mask_c1)
column_centroid = np.array([int(M_c1["m10"]/M_c1["m00"]), int(M_c1["m01"]/M_c1["m00"])])

mask_c7 = np.where(mask == target_class_id, 255, 0).astype(np.uint8)

kernel = np.ones((5, 5), np.uint8)
clean_mask = cv2.morphologyEx(mask_c7, cv2.MORPH_OPEN, kernel)
contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

max_dist, min_dist = 0, float('inf')
farthest_point, closest_point = None, None
for pt in cnt:
    p = pt[0]
    dist = np.linalg.norm(p - column_centroid)
    if dist > max_dist: max_dist, farthest_point = tuple(p)
    if dist < min_dist: min_dist, closest_point = tuple(p)

base_np = np.array(closest_point, dtype=float)
tip_np = np.array(farthest_point, dtype=float)
axis_vector = tip_np - base_np
true_length = np.linalg.norm(axis_vector)

unit_axis = axis_vector / true_length
perp_vector = np.array([-unit_axis[1], unit_axis[0]])

max_pos_proj, max_neg_proj = 0, 0
pos_pt, neg_pt = base_np, base_np 

for pt in cnt:
    p = np.array(pt[0], dtype=float)
    proj_dist = np.dot(p - base_np, perp_vector)
    if proj_dist > max_pos_proj: 
        max_pos_proj = proj_dist
        pos_pt = p
    elif proj_dist < max_neg_proj: 
        max_neg_proj = proj_dist
        neg_pt = p

true_width = max_pos_proj - max_neg_proj 
aspect_ratio = true_length / true_width if true_width > 0 else 0

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_facecolor('#E7E6E6')

cnt_pts = np.vstack([cnt[:, 0, :], cnt[0, 0, :]])
ax.plot(cnt_pts[:, 0], cnt_pts[:, 1], color='#333333', linewidth=2, zorder=2)
ax.plot([base_np[0], tip_np[0]], [base_np[1], tip_np[1]], color='royalblue', linewidth=2, linestyle='--', zorder=3)
ax.scatter(base_np[0], base_np[1], color='crimson', s=50, zorder=5, edgecolors='white', label='Base')
ax.scatter(tip_np[0], tip_np[1], color='seagreen', s=50, zorder=5, edgecolors='white', label='Tip')

proj_pos_axis = base_np + np.dot(pos_pt - base_np, unit_axis) * unit_axis
proj_neg_axis = base_np + np.dot(neg_pt - base_np, unit_axis) * unit_axis

ax.plot([pos_pt[0], proj_pos_axis[0]], [pos_pt[1], proj_pos_axis[1]], color='darkorange', linewidth=2, linestyle=':', zorder=3)
ax.plot([neg_pt[0], proj_neg_axis[0]], [neg_pt[1], proj_neg_axis[1]], color='darkorange', linewidth=2, linestyle=':', zorder=3)
ax.scatter([pos_pt[0], neg_pt[0]], [pos_pt[1], neg_pt[1]], color='darkorange', s=40, zorder=5, edgecolors='white')

ax.set_title(f"Test\nAR: {aspect_ratio:.3f}", fontsize=11, fontweight='bold')
ax.axis('equal')
ax.axis('off')
plt.savefig('packages/Quantification/test_plot.png')
print("Plot saved.")
