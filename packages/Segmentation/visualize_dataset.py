"""
Visualize dataset: original image + colorized mask overlay side by side.
Outputs a grid figure for train and val splits.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────
DATASET_ROOT = "./Datasets/dataset_cropped_19_5classes_LandR"
OUTPUT_DIR = "./Datasets/dataset_visualizations"
SPLITS = ["train", "val"]

CLASS_COLORS = {
    0: (0, 0, 0),        # Background
    1: (255, 255, 0),    # Column
    2: (0, 255, 0),      # Dorsal Sepal
    3: (0, 0, 255),      # Labellum
    4: (0, 255, 255),    # Lateral Sepal
    5: (255, 0, 0),      # Petal
    6: (255, 0, 255),    # Petal_L
    7: (255, 165, 0),    # Petal_R
}

CLASS_NAMES = [
    "Background", "Column", "Dorsal Sepal", "Labellum",
    "Lateral Sepal", "Petal", "Petal_L", "Petal_R",
]

OVERLAY_ALPHA = 0.45


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a single-channel class-index mask to RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS.items():
        rgb[mask == cls_idx] = color
    return rgb


def make_legend_patches():
    """Create legend patches for all classes."""
    patches = []
    for idx, name in enumerate(CLASS_NAMES):
        color = np.array(CLASS_COLORS[idx]) / 255.0
        patches.append(mpatches.Patch(color=color, label=f"{idx}: {name}"))
    return patches


def visualize_split(split: str):
    img_dir = os.path.join(DATASET_ROOT, split, "images")
    mask_dir = os.path.join(DATASET_ROOT, split, "masks")

    # Collect valid image-mask pairs
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png")) and f != "Thumbs.db"
    ])

    pairs = []
    for img_name in img_files:
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            pairs.append((img_name, mask_name))

    if not pairs:
        print(f"[{split}] No image-mask pairs found, skipping.")
        return

    n = len(pairs)
    print(f"[{split}] Found {n} image-mask pairs")

    # Layout: each pair gets 3 columns (original, mask, overlay)
    cols = 3
    fig, axes = plt.subplots(n, cols, figsize=(cols * 5, n * 4.5))
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2D

    fig.suptitle(f"Dataset: {split} split ({n} samples)", fontsize=16, fontweight="bold", y=1.0)

    for row, (img_name, mask_name) in enumerate(pairs):
        # Load
        img = np.array(Image.open(os.path.join(img_dir, img_name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(mask_dir, mask_name)).convert("L"))

        # Colorize mask
        mask_rgb = colorize_mask(mask)

        # Overlay
        overlay = (img.astype(np.float32) * (1 - OVERLAY_ALPHA)
                    + mask_rgb.astype(np.float32) * OVERLAY_ALPHA).astype(np.uint8)

        # Unique classes in this mask
        present = sorted(np.unique(mask))
        class_str = ", ".join([f"{c}:{CLASS_NAMES[c]}" for c in present if c < len(CLASS_NAMES)])

        # Plot
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{img_name}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_rgb)
        axes[row, 1].set_title(f"Mask — classes: {class_str}", fontsize=7)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay", fontsize=8)
        axes[row, 2].axis("off")

    # Add shared legend
    legend_patches = make_legend_patches()
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(CLASS_NAMES),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"dataset_overview_{split}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{split}] Saved → {save_path}")


def main():
    for split in SPLITS:
        visualize_split(split)
    print("Done.")


if __name__ == "__main__":
    main()
