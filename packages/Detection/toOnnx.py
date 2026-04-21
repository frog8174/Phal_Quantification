from ultralytics import YOLO

model = YOLO('flower_detect_v3_1types/FD/weights/best.pt')
# model = YOLO(r'/home/nas2/Workspace/Aaron/Allen_model/best-det20.pt')

model.export(
            format="onnx", 
            half=False, 
            imgsz = 960)  # or format="onnx"
# /home/nas2/Workspace/Aaron/yolov12/TNC_detect_rpi2_L/enhanced_20_1920p_v12m3