# Pseudo-Labeling Pipeline

> **YOLO → SPTS → SAM2 → DINOv3 + Linear Head → Pseudo Mask**

自動化的端對端 Pseudo Label 產生管線。給定一批未標註的蘭花照片，自動產出 8-class semantic segmentation mask，可直接回餵至 `fine-tuning.py` 進行下一輪半監督學習。

## Pipeline 架構

```
Raw Image
    │
    ▼
[1] YOLO Flower Detection ──── 偵測所有花朵 BBox
    │
    ▼
[2] SPTS (Spatial Prominence) ── 選出主花 (面積最大 + 最靠近中心)
    │
    ▼
[3] SAM2 Instance Segmentation ── 以 BBox 為 prompt 產出前景 binary mask
    │
    ▼
[4] Crop & Black-BG ──────────── 裁切主花 + 黑底化
    │
    ▼
[5] DINOv3 + Linear Head ────── Fine-tuned 模型推論 8-class semantic mask
    │
    ▼
[6] Output ────────────────────── pseudo_masks/ + visualization/
```

## 目錄結構

```
PseudoLabel/
├── README.md               ← 本文件
├── config.py               ← 全域設定 (路徑、裝置、閾值)
├── pipeline.py             ← 主管線腳本 (端對端)
├── inputs/                 ← 放入未標註的原始照片
│   └── *.jpg / *.png
├── outputs/
│   ├── crops/              ← SAM2 裁切後的主花 (黑底 RGB)
│   ├── pseudo_masks/       ← DINOv3 產出的 8-class mask (uint8 grayscale)
│   └── visualization/      ← 三合一視覺化 (原圖 / mask / overlay)
└── logs/
    └── pipeline_log.csv    ← 每張圖的處理紀錄
```

## 使用方式

```bash
# 1. 把未標註照片放入 inputs/
# 2. 確認 config.py 的路徑與裝置設定
# 3. 執行
cd packages/PseudoLabel
uv run pipeline.py
```

## 產出物用途

| 產出 | 用途 |
|------|------|
| `pseudo_masks/*.png` | 回餵至 `Segmentation/fine-tuning.py` 作為額外訓練資料 |
| `crops/*.png` | 可直接餵入 `Quantification/ColorQuantify.ipynb` 做色素分析 |
| `visualization/*.png` | 人工 QC 檢查 pseudo label 品質 |

## 依賴

- `ultralytics` (YOLO)
- `sam2` (SAM2)
- `transformers` (DINOv3)
- `torch`, `torchvision`, `opencv-python`, `numpy`, `pandas`
