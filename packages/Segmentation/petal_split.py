"""
Petal L/R Post-processing Splitter
====================================
Given a 6-class semantic mask (with merged Petal), split the Petal region
into Petal_L and Petal_R using the Column centroid as the symmetry axis.

6-class mapping:
  0: Background, 1: Column, 2: Dorsal Sepal,
  3: Labellum, 4: Lateral Sepal, 5: Petal (merged)

8-class output mapping:
  0: Background, 1: Column, 2: Dorsal Sepal,
  3: Labellum, 4: Lateral Sepal, 5: (unused),
  6: Petal_L, 7: Petal_R
"""

import numpy as np
import cv2


# ── Class indices ──
CLASS_COLUMN = 1
CLASS_PETAL_MERGED = 5
CLASS_PETAL_L = 6
CLASS_PETAL_R = 7


def get_column_centroid(mask, column_class=CLASS_COLUMN):
    """Get the centroid (cx, cy) of the Column region.
    Returns None if Column is not found.
    """
    column_mask = (mask == column_class).astype(np.uint8)
    if column_mask.sum() == 0:
        return None

    moments = cv2.moments(column_mask)
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def get_flower_centroid(mask):
    """Fallback: get centroid of all non-background pixels."""
    flower_mask = (mask > 0).astype(np.uint8)
    if flower_mask.sum() == 0:
        return None

    moments = cv2.moments(flower_mask)
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def split_petal(mask_6class):
    """Split merged Petal (class 5) into Petal_L (6) and Petal_R (7).

    Strategy:
      1. Find Column centroid x-coordinate as the symmetry axis
      2. Petal pixels left of axis → Petal_L (6)
      3. Petal pixels right of axis → Petal_R (7)
      4. Fallback to flower centroid if Column not found

    Args:
        mask_6class: (H, W) uint8 array with 6-class values (0-5)

    Returns:
        mask_8class: (H, W) uint8 array with 8-class values
        centroid_x: the x coordinate used as the split axis
        method: 'column' or 'flower_fallback' or 'none'
    """
    mask_8class = mask_6class.copy()

    # Find split axis
    centroid = get_column_centroid(mask_8class)
    method = "column"

    if centroid is None:
        centroid = get_flower_centroid(mask_8class)
        method = "flower_fallback"

    if centroid is None:
        # No flower found at all
        return mask_8class, None, "none"

    cx = centroid[0]

    # Split petal region
    petal_mask = mask_8class == CLASS_PETAL_MERGED
    petal_left = petal_mask & (np.arange(mask_8class.shape[1])[None, :] < cx)
    petal_right = petal_mask & (np.arange(mask_8class.shape[1])[None, :] >= cx)

    mask_8class[petal_left] = CLASS_PETAL_L
    mask_8class[petal_right] = CLASS_PETAL_R

    return mask_8class, cx, method


def merge_petal_to_6class(mask_8class):
    """Convert 8-class mask to 6-class by merging Petal_L (6) + Petal_R (7) → Petal (5).

    Also handles the case where class 5 was unused in the original 8-class scheme.
    """
    mask_6class = mask_8class.copy()
    mask_6class[mask_6class == CLASS_PETAL_L] = CLASS_PETAL_MERGED
    mask_6class[mask_6class == CLASS_PETAL_R] = CLASS_PETAL_MERGED
    return mask_6class
