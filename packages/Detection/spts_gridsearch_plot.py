"""
Step 2 (需 matplotlib，不需 ultralytics):讀 boxes_cache.json，把 (alpha, beta)
在 0..3 / step 0.5 的 7x7 網格全算一遍準確率，輸出:
  - spts_eval_outputs/grid_accuracy.csv       完整 7x7 表
  - spts_eval_outputs/spts_gridsearch_heatmap.png   2D 熱圖(主圖)
  - spts_eval_outputs/spts_alpha_barchart.png       beta=1.5 切片長條圖(口試輔助)

準確率定義同 evaluate_spts.py:預測主花與 GT 的 IoU > 0.5 即為選對。
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
CACHE = D / "spts_eval_outputs" / "boxes_cache.json"
GT_FILE = D / "primary_flower_gt.json"
OUTDIR = D / "spts_eval_outputs"

GRID = np.arange(0.0, 3.01, 0.5)          # 0,0.5,...,3.0  (7 值)
BETA_SLICE = 1.5                          # 長條圖固定的 beta
DEFAULT = (1.0, 1.5)                      # 預設權重
ADOPTED = (2.0, 1.5)                      # 論文採用的 optimal(高原上的代表點)
IOU_THR = 0.5

cache = json.load(open(CACHE))
gt = json.load(open(GT_FILE, encoding="utf-8"))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def accuracy(alpha, beta):
    correct = total = 0
    for name, g in gt.items():
        rec = cache.get(name)
        if not rec or not rec["boxes"]:
            continue
        h, w = rec["h"], rec["w"]
        boxes = rec["boxes"]
        max_area = max((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
        diag = np.hypot(w/2, h/2)
        best, bs = None, -1e9
        for b in boxes:
            na = ((b[2]-b[0])*(b[3]-b[1])) / max_area if max_area > 0 else 0
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            nd = np.hypot(cx - w/2, cy - h/2) / diag
            s = alpha*na - beta*nd
            if s > bs:
                bs, best = s, b
        total += 1
        correct += iou(best, g) > IOU_THR
    return correct / total * 100 if total else 0.0, correct, total


# ── 全網格 ──
acc = np.zeros((len(GRID), len(GRID)))
for i, a in enumerate(GRID):
    for j, b in enumerate(GRID):
        acc[i, j], _, N = accuracy(a, b)

acc_max = acc.max()
plateau = np.argwhere(np.abs(acc - acc_max) < 1e-9)   # 所有達到最高的格
di = int(np.argmin(np.abs(GRID - DEFAULT[0]))); dj = int(np.argmin(np.abs(GRID - DEFAULT[1])))
ai = int(np.argmin(np.abs(GRID - ADOPTED[0]))); aj = int(np.argmin(np.abs(GRID - ADOPTED[1])))

# CSV
import csv
with open(OUTDIR / "grid_accuracy.csv", "w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["alpha\\beta"] + [f"{b:.1f}" for b in GRID])
    for i, a in enumerate(GRID):
        wtr.writerow([f"{a:.1f}"] + [f"{acc[i, j]:.2f}" for j in range(len(GRID))])

print(f"N = {N} images")
print(f"Max accuracy = {acc_max:.2f}%  achieved by {len(plateau)} (alpha,beta) combos (plateau)")
print(f"Adopted: alpha={ADOPTED[0]}, beta={ADOPTED[1]} -> {acc[ai, aj]:.2f}%")
print(f"Default: alpha={DEFAULT[0]}, beta={DEFAULT[1]} -> {acc[di, dj]:.2f}%")

# ── 熱圖 ──
fig, ax = plt.subplots(figsize=(7.2, 6), facecolor="white")
im = ax.imshow(acc, origin="lower", cmap="viridis", aspect="auto")
ax.set_xticks(range(len(GRID))); ax.set_xticklabels([f"{b:.1f}" for b in GRID])
ax.set_yticks(range(len(GRID))); ax.set_yticklabels([f"{a:.1f}" for a in GRID])
ax.set_xlabel(r"$\beta$  (distance-penalty weight)", fontsize=12)
ax.set_ylabel(r"$\alpha$  (area-reward weight)", fontsize=12)
ax.set_title("Primary-flower selection accuracy over the (α, β) grid", fontsize=12.5, pad=30)
for i in range(len(GRID)):
    for j in range(len(GRID)):
        ax.text(j, i, f"{acc[i, j]:.1f}", ha="center", va="center",
                color="white" if acc[i, j] < acc.max()-4 else "black", fontsize=7.5)
# 整片高原(達最高值的所有格)= 細紅框
for (i, j) in plateau:
    ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False,
                 edgecolor="red", lw=1.3))
# 論文採用點 = 粗紅框
ax.add_patch(plt.Rectangle((aj-.5, ai-.5), 1, 1, fill=False, edgecolor="red", lw=3))
# 預設 = 白虛框
ax.add_patch(plt.Rectangle((dj-.5, di-.5), 1, 1, fill=False,
             edgecolor="white", lw=2.0, ls="--"))
ax.text(aj, ai+.34, "adopted", ha="center", color="red", fontsize=8, fontweight="bold")
ax.text(dj, di-.40, "default", ha="center", color="white", fontsize=8)
ax.text(0.5, 1.015, f"Top accuracy {acc_max:.2f}% forms a plateau of {len(plateau)} "
        f"(α, β) settings (thin red); the study adopted (2.0, 1.5)",
        transform=ax.transAxes, ha="center", fontsize=9, color="#444")
cb = fig.colorbar(im, ax=ax); cb.set_label("Accuracy (%)")
fig.tight_layout()
fig.savefig(OUTDIR / "spts_gridsearch_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ── 長條圖:固定 beta=1.5,看 alpha ──
row = [accuracy(a, BETA_SLICE)[0] for a in GRID]
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
bars = ax.bar([f"{a:.1f}" for a in GRID], row, color="#7aa6c2", edgecolor="#33566b")
kbest = int(np.argmax(row))
bars[kbest].set_color("#e07a5f"); bars[kbest].set_edgecolor("#9c3d28")
for i, v in enumerate(row):
    ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9,
            fontweight="bold" if i == kbest else "normal")
ax.set_xlabel(r"$\alpha$  (area-reward weight),  with $\beta$ fixed at 1.5", fontsize=12)
ax.set_ylabel("Selection accuracy (%)", fontsize=12)
ax.set_title(r"Accuracy vs. $\alpha$ (distance weight $\beta$ = 1.5)", fontsize=12.5)
ax.set_ylim(min(row) - 4, max(row) + 4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUTDIR / "spts_alpha_barchart.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: grid_accuracy.csv, spts_gridsearch_heatmap.png, spts_alpha_barchart.png -> {OUTDIR}")
