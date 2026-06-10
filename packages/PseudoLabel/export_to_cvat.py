"""
Export Pseudo Masks to CVAT Segmentation Mask 1.1 Format
==========================================================
讀取 uint8 index 格式的 pseudo masks，根據 config 轉換為 RGB 色彩，
並產生 CVAT 支援的 labelmap.txt，最後打包成 zip。
"""

import os
import cv2
import shutil
import zipfile
import numpy as np
from pathlib import Path

# pyrefly: ignore [missing-import]
from config import MASK_DIR, CLASS_NAMES, CLASS_COLORS


def export_to_cvat(output_zip="cvat_pseudo_labels.zip"):
    export_dir = Path("./cvat_export_temp")
    seg_class_dir = export_dir / "SegmentationClass"
    image_sets_dir = export_dir / "ImageSets" / "Segmentation"
    
    # 建立暫存資料夾
    if export_dir.exists():
        shutil.rmtree(export_dir)
    seg_class_dir.mkdir(parents=True)
    image_sets_dir.mkdir(parents=True)

    # 1. 產生 labelmap.txt
    labelmap_path = export_dir / "labelmap.txt"
    with open(labelmap_path, "w", encoding="utf-8") as f:
        f.write("# label:color_rgb:parts:actions\n")
        for i, name in enumerate(CLASS_NAMES):
            r, g, b = CLASS_COLORS[i]
            # CVAT 要求背景必須嚴格小寫 "background"，否則會被當作自定義標籤
            export_name = "background" if name.lower() == "background" else name
            # CVAT format: class_name:R,G,B::
            f.write(f"{export_name}:{r},{g},{b}::\n")
    print(f"Created labelmap.txt with {len(CLASS_NAMES)} classes.")

    # 2. 轉換 Masks 為 RGB
    mask_files = list(Path(MASK_DIR).glob("*.png"))
    if not mask_files:
        print(f"No masks found in {MASK_DIR}.")
        return

    print(f"Converting {len(mask_files)} masks to RGB format...")
    for mask_path in mask_files:
        # 讀取 uint8 index mask
        mask_idx = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_idx is None:
            continue
        
        # 建立 RGB 空白畫布
        h, w = mask_idx.shape
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # 根據 CLASS_COLORS 填色 (OpenCV 預設是 BGR，但我們存 RGB 給 CVAT)
        for i in range(len(CLASS_NAMES)):
            r, g, b = CLASS_COLORS[i]
            mask_rgb[mask_idx == i] = [r, g, b]

        # 寫入 SegmentationClass 目錄 (記得轉回 BGR 讓 cv2.imwrite 存出正確顏色)
        out_path = seg_class_dir / mask_path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

    # 3. 產生 ImageSets/Segmentation/default.txt
    default_txt_path = image_sets_dir / "default.txt"
    with open(default_txt_path, "w", encoding="utf-8") as f:
        for mask_path in mask_files:
            f.write(f"{mask_path.stem}\n")
    print("Created ImageSets/Segmentation/default.txt")

    # 4. 打包成 ZIP
    print(f"Zipping to {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 加入 labelmap.txt
        zipf.write(labelmap_path, arcname="labelmap.txt")
        # 加入 ImageSets
        zipf.write(default_txt_path, arcname="ImageSets/Segmentation/default.txt")
        # 加入所有 masks
        for f in seg_class_dir.iterdir():
            zipf.write(f, arcname=f"SegmentationClass/{f.name}")

    # 清理暫存
    shutil.rmtree(export_dir)
    print("Done! You can now import this zip into CVAT using 'Segmentation Mask 1.1'.")

if __name__ == "__main__":
    export_to_cvat()
