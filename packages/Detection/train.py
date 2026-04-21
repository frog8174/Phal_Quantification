
import os
from ultralytics import YOLO

def main():
    project_name = 'flower_detect'
    
    model = YOLO("yolo11n.pt")  

    results  = model.train(
        data="datasets/flower.yaml",        # COCO‑format dataset
        project = project_name,
        workers = 8,
        name = 'FD',
        epochs=200,
        patience=100,
        imgsz=960,                # 放大解析度 → 讓小物體更明顯
        batch=8,                  # 視 GPU RAM 調整
        device='2,3',
        optimizer="SGD",        # SGD 或 AdamW 由框架自選
        lr0=0.003,
        lrf=0.00005,
        cache=False,
        amp=True,
        augment=True,
        dropout=0.1,
        # ----------------Augment-----------------------------
        mosaic=0.45,               # 👈 50% 批次做 Mosaic，避免框數爆增
        close_mosaic=30,          # 👈 最後 10 個 epoch 關閉 Mosaic
        degrees = 0,
        shear = 10,
        flipud = 0.1,
        fliplr = 0.1,
        hsv_h = 0.02,
        hsv_s = 0.3,
        hsv_v = 0.15,
        translate = 0,
        
    )

    # best_pt = model.trainer.save_dir / "weights" / "best.pt"
    # exp.log_model("best_weights", file_or_folder=str(best_pt))
    # exp.end()   # optional：標記 experiment 結束

if __name__ == "__main__":
    main()

