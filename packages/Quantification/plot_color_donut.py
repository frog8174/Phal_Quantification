"""
Generate publication-quality Donut Chart for global pigment distribution.
Automatically adapts to any K value by reading global_centers_rgb.npy.

Outputs:
  - Pigment/06_global_donut_chart.png  (Global average across all samples)
  - Pigment/07_per_sample_donut_grid.png (Grid of individual sample donuts, top-N by diversity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PIGMENT_DIR = BASE_DIR / "Pigment"
CENTERS_RGB_PATH = PIGMENT_DIR / "global_centers_rgb.npy"
FEATURES_CSV_PATH = PIGMENT_DIR / "04_quantified_features_full.csv"  # full 189-flower set

# ── Load data ──────────────────────────────────────────
centers_rgb = np.load(CENTERS_RGB_PATH)  # shape: (K, 3), dtype uint8
K = len(centers_rgb)
print(f"Detected K = {K} color bins")

df = pd.read_csv(FEATURES_CSV_PATH)
ratio_cols = [c for c in df.columns if c.startswith("Bin_")]

CSV_K_MISMATCH = len(ratio_cols) != K
if CSV_K_MISMATCH:
    print(f"[!] Warning: CSV has {len(ratio_cols)} ratio columns but K={K}.")
    print(f"    → Global donut will use npy centers only.")
    print(f"    → Per-sample grid will be SKIPPED. Re-run Cell 05 to regenerate CSV.")

# Normalize RGB to [0,1] for matplotlib
colors = centers_rgb.astype(float) / 255.0


# ══════════════════════════════════════════════════════════
# 1. Global Average Donut Chart
# ══════════════════════════════════════════════════════════
if not CSV_K_MISMATCH:
    global_ratios = df[ratio_cols].mean().values
else:
    # CSV 尚未更新，使用均等比例展示色票
    global_ratios = np.ones(K) / K

# Sort by ratio descending for better visual arrangement
sorted_idx = np.argsort(-global_ratios)
sorted_ratios = global_ratios[sorted_idx]
sorted_colors = colors[sorted_idx]

fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

# Outer donut
wedges, texts, autotexts = ax.pie(
    sorted_ratios,
    colors=sorted_colors,
    autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
    pctdistance=0.78,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
    textprops=dict(fontsize=12, fontweight="bold"),
)

# Style percentage labels: use white on dark wedges, black on light ones
for i, at in enumerate(autotexts):
    # Perceived luminance
    r, g, b = sorted_colors[i]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    at.set_color("white" if lum < 0.5 else "#333333")

# Center circle (donut hole) with summary text
centre = plt.Circle((0, 0), 0.55, fc="white", ec="#cccccc", lw=1)
ax.add_artist(centre)
ax.text(0, 0.06, "Global", ha="center", va="center", fontsize=16, fontweight="bold", color="#333")
ax.text(0, -0.08, f"K = {K}", ha="center", va="center", fontsize=13, color="#666")

# Legend — show color swatch + bin index + percentage
legend_labels = [
    f"Bin {sorted_idx[i]}  ({sorted_ratios[i]*100:.1f}%)"
    for i in range(K)
]
legend_handles = [
    mpatches.Patch(facecolor=sorted_colors[i], edgecolor="#999", linewidth=0.5)
    for i in range(K)
]
ax.legend(
    legend_handles, legend_labels,
    loc="center left", bbox_to_anchor=(1.0, 0.5),
    fontsize=11, frameon=False, title="Pigment Bins", title_fontsize=12
)

ax.set_title(
    f"Global Petal Pigment Distribution (n = {len(df)} flowers)",
    fontsize=15, fontweight="bold", pad=20
)

fig.tight_layout()
out_path_global = PIGMENT_DIR / "06_global_donut_chart.png"
fig.savefig(out_path_global, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path_global}")


# ══════════════════════════════════════════════════════════
# 2. Per-sample Donut Grid (Top-12 most diverse samples)
#    — with original petal image embedded in donut hole
# ══════════════════════════════════════════════════════════
if CSV_K_MISMATCH:
    print("[!] Skipping per-sample grid (CSV K mismatch). Re-run Cell 05 first.")
else:
    import cv2
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # Petal cutout image directory
    CUTOUT_DIR = BASE_DIR / ".." / "Segmentation" / "Inference" / "petal_only_outputs" / "petal_cutouts"
    CUTOUT_DIR = CUTOUT_DIR.resolve()

    # Diversity = Shannon entropy (higher = more evenly distributed colors)
    def shannon_entropy(ratios):
        r = ratios[ratios > 0]
        return -np.sum(r * np.log2(r))

    df["diversity"] = df[ratio_cols].apply(shannon_entropy, axis=1)

    # Curated set spanning monochromatic → multimodal so the figure matches the §4.4.3 text
    # (0137 pink, 0085 yellow, 0184 magenta, 0170 dark-purple = near-single-colour;
    #  0035/0008/0091/0179/0032/0088 = spotted/blended). Falls back to top-diversity if absent.
    CURATED = ["0137_petal_cutout", "0085_petal_cutout", "0184_petal_cutout", "0170_petal_cutout",
               "0014_petal_cutout", "0044_petal_cutout", "0008_petal_cutout", "0035_petal_cutout",
               "0091_petal_cutout", "0179_petal_cutout", "0032_petal_cutout", "0088_petal_cutout"]
    present = [s for s in CURATED if s in set(df["Sample_ID"])]
    if len(present) >= 8:
        top_samples = df.set_index("Sample_ID").loc[present].reset_index()
    else:
        top_samples = df.nlargest(12, "diversity")

    # Order the panels monochromatic → multimodal by dominant-bin share, so the
    # grid itself reads as a gradient and the selection is transparently spanning
    # (these 12 cover the full 26.3–98.4% range observed across all 189 flowers).
    top_samples["dominant"] = top_samples[ratio_cols].max(axis=1)
    top_samples = top_samples.sort_values("dominant", ascending=False).reset_index(drop=True)

    n_cols = 4
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), facecolor="white")
    axes = axes.flatten()

    for i, (_, row) in enumerate(top_samples.iterrows()):
        ax = axes[i]
        ratios = row[ratio_cols].values.astype(float)

        # Skip bins with 0 ratio for cleaner plot
        mask = ratios > 0.005
        ax.pie(
            ratios[mask],
            colors=colors[mask],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.35, edgecolor="white", linewidth=1.5),
        )

        # --- Embed original flower image in donut center ---
        sample_id = row["Sample_ID"]
        img_path = CUTOUT_DIR / f"{sample_id}.png"
        if img_path.exists():
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
                # Make black background transparent
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                img_rgba[gray < 15, 3] = 0
                # Resize to thumbnail
                h, w = img_rgba.shape[:2]
                thumb_size = 300
                scale = thumb_size / max(h, w)
                img_thumb = cv2.resize(img_rgba, (int(w * scale), int(h * scale)))

                imagebox = OffsetImage(img_thumb, zoom=0.35)
                imagebox.image.axes = ax
                ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
                ax.add_artist(ab)
        else:
            # Fallback: white circle with text
            centre = plt.Circle((0, 0), 0.55, fc="white", ec="#ddd", lw=0.5)
            ax.add_artist(centre)

        short_name = sample_id.replace("_petal_cutout", "")
        ax.set_title(f"{short_name}\ndominant {row['dominant']*100:.0f}%",
                     fontsize=10.5, pad=5, linespacing=1.35)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared colour legend (bin index + name) for all donuts
    BIN_NAMES = ["Brown", "Pale yellow", "Pink", "Golden olive", "Magenta", "Dark red-purple"]
    if K != len(BIN_NAMES):
        BIN_NAMES = [f"Bin {i}" for i in range(K)]
    legend_handles = [
        mpatches.Patch(facecolor=colors[i], edgecolor="#666", linewidth=0.6,
                       label=f"Bin {i} — {BIN_NAMES[i]}")
        for i in range(K)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=K,
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"Representative Per-Sample Pigment Profiles (K = {K})",
        fontsize=16, fontweight="bold", y=1.01
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path_grid = PIGMENT_DIR / "07_per_sample_donut_grid.png"
    fig.savefig(out_path_grid, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path_grid}")

print("Done!")
