"""
Figure 13 — Global pigment codebook (K = 6).

Renders the six K-Means cluster centers as a table with an inline color swatch
column, so that each swatch sits on the same row as its CIELAB coordinates.

Note: global_centers_lab.npy stores OpenCV 8-bit LAB (L in 0-255, a/b offset by
128), NOT CIELAB. It is converted here before plotting.

Output: PaperWriting/figures/figure13_palette_table.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PIGMENT_DIR = BASE_DIR / "Pigment"
OUT_PATH = BASE_DIR / ".." / ".." / "PaperWriting" / "figures" / "figure13_palette_table.png"
OUT_PATH = OUT_PATH.resolve()

NAMES = ["Brown", "Pale yellow", "Pink", "Golden olive", "Magenta", "Dark red-purple"]

# ── Load and convert OpenCV 8-bit LAB → CIELAB ─────────────
lab_cv = np.load(PIGMENT_DIR / "global_centers_lab.npy")
rgb = np.load(PIGMENT_DIR / "global_centers_rgb.npy")

L = lab_cv[:, 0] * 100.0 / 255.0
a = lab_cv[:, 1] - 128.0
b = lab_cv[:, 2] - 128.0
C = np.sqrt(a**2 + b**2)

# ── Table layout ───────────────────────────────────────────
# (label, x_left, x_right, alignment)
COLS = [
    ("",      0.000, 0.105, "c"),   # swatch
    ("Bin",   0.105, 0.185, "c"),
    ("Name",  0.185, 0.450, "l"),
    ("$L^*$", 0.450, 0.560, "r"),
    ("$a^*$", 0.560, 0.670, "r"),
    ("$b^*$", 0.670, 0.780, "r"),
    ("$C^*$", 0.780, 0.880, "r"),
    ("Hex",   0.880, 1.000, "c"),
]

N = len(NAMES)
ROW_H = 1.0 / (N + 1)          # +1 header row
PAD = 0.018                    # inner cell padding for l/r alignment
GRID = "#b0b0b0"
FS = 15

fig, ax = plt.subplots(figsize=(11, 3.6), facecolor="white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def cell_text(text, col, y_center, weight="normal", color="#1a1a1a", size=FS):
    _, x0, x1, align = col
    if align == "l":
        x, ha = x0 + PAD, "left"
    elif align == "r":
        x, ha = x1 - PAD, "right"
    else:
        x, ha = (x0 + x1) / 2, "center"
    ax.text(x, y_center, text, ha=ha, va="center",
            fontsize=size, fontweight=weight, color=color)


# ── Header ─────────────────────────────────────────────────
y_head = 1.0 - ROW_H / 2
ax.add_patch(Rectangle((0, 1.0 - ROW_H), 1.0, ROW_H,
                       facecolor="#eeeeee", edgecolor=GRID, linewidth=0.9))
for col in COLS:
    if col[0]:
        cell_text(col[0], col, y_head, weight="bold")

# ── Body rows ──────────────────────────────────────────────
for i in range(N):
    y_top = 1.0 - ROW_H * (i + 1)
    y_c = y_top - ROW_H / 2

    # zebra striping for readability
    if i % 2 == 1:
        ax.add_patch(Rectangle((0, y_top - ROW_H), 1.0, ROW_H,
                               facecolor="#fafafa", edgecolor="none", zorder=0))

    # swatch cell — inset so the fill reads as a chip, not a full-bleed band
    sw_x0, sw_x1 = COLS[0][1], COLS[0][2]
    inset_x, inset_y = 0.012, ROW_H * 0.18
    ax.add_patch(Rectangle((sw_x0 + inset_x, y_top - ROW_H + inset_y),
                           (sw_x1 - sw_x0) - 2 * inset_x, ROW_H - 2 * inset_y,
                           facecolor=rgb[i] / 255.0,
                           edgecolor="#555555", linewidth=0.8, zorder=2))

    cell_text(str(i), COLS[1], y_c)
    cell_text(NAMES[i], COLS[2], y_c)
    cell_text(f"{L[i]:.1f}", COLS[3], y_c)
    cell_text(f"{a[i]:+.1f}".replace("-", "−"), COLS[4], y_c)
    cell_text(f"{b[i]:+.1f}".replace("-", "−"), COLS[5], y_c)
    cell_text(f"{C[i]:.1f}", COLS[6], y_c)
    cell_text("#{:02X}{:02X}{:02X}".format(*rgb[i]), COLS[7], y_c,
              color="#555555", size=FS - 2.5)

# ── Rules: top, under header, bottom (booktabs style) ──────
for y, lw in [(1.0, 1.4), (1.0 - ROW_H, 0.9), (1.0 - ROW_H * (N + 1), 1.4)]:
    ax.plot([0, 1], [y, y], color="#333333", linewidth=lw, zorder=3)

fig.tight_layout(pad=0.4)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
