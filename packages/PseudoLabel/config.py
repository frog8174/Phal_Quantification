"""
Global configuration for the Pseudo-Labeling pipeline.
All paths and hyperparameters are centralized here.
"""

import torch

# ==========================================
# Device
# ==========================================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# ==========================================
# Paths
# ==========================================
# Input: unlabeled raw images
INPUT_DIR = "./inputs"

# Output directories
OUTPUT_DIR = "./outputs"
CROP_DIR = f"{OUTPUT_DIR}/crops"
MASK_DIR = f"{OUTPUT_DIR}/pseudo_masks"
VIZ_DIR = f"{OUTPUT_DIR}/visualization"
STEPS_DIR = f"{OUTPUT_DIR}/pipeline_steps"  # 每張圖的逐步驟視覺化
LOG_DIR = "./logs"

# ==========================================
# Stage 1: YOLO Flower Detection
# ==========================================
YOLO_WEIGHTS = "../Detection/training_results/flower_detect/FD11/weights/best.pt"
YOLO_IMG_SIZE = 960
YOLO_CONF = 0.5
YOLO_IOU = 0.75
YOLO_MAX_DET = 100

# SPTS (Spatial Prominence-based Target Selection)
SPTS_ALPHA = 1.0   # area weight
SPTS_BETA = 1.0    # distance penalty

# ==========================================
# Stage 2: SAM2 Instance Segmentation
# ==========================================
SAM2_MODEL = "facebook/sam2-hiera-large"

# ==========================================
# Stage 3: DINOv3 Semantic Segmentation
# ==========================================
DINO_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_CHECKPOINT = "../Segmentation/training_result/Finalexp_lastlayer_lr2e-4_v1/best_finetune_LandR.pth"
DINO_IMG_SIZE = 2400
DINO_PATCH_SIZE = 16
DINO_NUM_CLASSES = 8
DINO_EXTRACT_LAYERS = [-1]

# ==========================================
# Class definitions (must match fine-tuning)
# ==========================================
CLASS_NAMES = [
    "Background", "Column", "Dorsal Sepal", "Labellum",
    "Lateral Sepal", "Petal", "Petal_L", "Petal_R"
]

CLASS_COLORS = {
    0: [0, 0, 0],       # Background
    1: [255, 255, 0],   # Column
    2: [0, 255, 0],     # Dorsal Sepal
    3: [0, 0, 255],     # Labellum
    4: [0, 255, 255],   # Lateral Sepal
    5: [255, 0, 0],     # Petal
    6: [255, 0, 255],   # Petal_L
    7: [255, 165, 0]    # Petal_R
}
