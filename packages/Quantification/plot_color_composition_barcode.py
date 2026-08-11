"""
Figure 14 — Pigment composition of all 189 flowers.

One column per flower, stacked by the six-bin codebook proportions. Within each
flower the segments are ordered by decreasing share, and the flowers themselves
are ordered by decreasing dominant share, so the monochromatic-to-multimodal
gradient of the whole collection is visible in a single panel.

Output: PaperWriting/figures/figure14_composition_barcode.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PIGMENT_DIR = BASE_DIR / "Pigment"
OUT_PATH = (BASE_DIR / ".." / ".." / "PaperWriting" / "figures"
            / "figure14_composition_barcode.png").resolve()

NAMES = ["Brown", "Pale yellow", "Pink", "Golden olive", "Magenta", "Dark red-purple"]
MONO_THRESHOLD = 0.70

centers_rgb = np.load(PIGMENT_DIR / "global_centers_rgb.npy")
colors = centers_rgb.astype(float) / 255.0

df = pd.read_csv(PIGMENT_DIR / "04_quantified_features_full.csv")
ratio_cols = [c for c in df.columns if c.startswith("Bin_")]
props = df[ratio_cols].values                      # (189, 6)
n = len(props)

# ── Order flowers by descending dominant share ─────────────
dominant = props.max(axis=1)
order = np.argsort(-dominant)
props = props[order]
dominant = dominant[order]
n_mono = int((dominant > MONO_THRESHOLD).sum())

fig, ax = plt.subplots(figsize=(13, 5.0), facecolor="white")

# ── Stack each flower's bins in decreasing-share order ─────
for i, row in enumerate(props):
    seg_order = np.argsort(-row)
    bottom = 0.0
    for b in seg_order:
        h = row[b]
        if h <= 0:
            continue
        ax.bar(i, h, bottom=bottom, width=1.0,
               color=colors[b], edgecolor="none")
        bottom += h

# ── Monochromatic / multimodal reference marks ─────────────
ax.axhline(MONO_THRESHOLD, color="#222222", linestyle="--", linewidth=1.4, zorder=5)
ax.axvline(n_mono - 0.5, color="#222222", linestyle="-", linewidth=1.4, zorder=5)

ax.text(n_mono / 2, 1.045,
        f"near-monochromatic\n{n_mono}/{n} ({n_mono/n*100:.0f}%)",
        ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#222")
ax.text(n_mono + (n - n_mono) / 2, 1.045,
        f"multimodal\n{n-n_mono}/{n} ({(n-n_mono)/n*100:.0f}%)",
        ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#222")
ax.text(n - 1.5, MONO_THRESHOLD + 0.015,
        f"dominant bin = {MONO_THRESHOLD:.0%}", ha="right", va="bottom",
        fontsize=10, style="italic", color="#222")

ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(0, 1.0)
ax.set_xlabel(f"Flower specimens, ordered by dominant-bin share  (n = {n})", fontsize=12.5)
ax.set_ylabel("Proportion of petal area", fontsize=12.5)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
ax.set_xticks([])
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

handles = [mpatches.Patch(facecolor=colors[i], edgecolor="#666", linewidth=0.6,
                          label=f"Bin {i} — {NAMES[i]}") for i in range(len(NAMES))]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
          ncol=6, frameon=False, fontsize=11)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT_PATH}  (n={n}, mono={n_mono})")
