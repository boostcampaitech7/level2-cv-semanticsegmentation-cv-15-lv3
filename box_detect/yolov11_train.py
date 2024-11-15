from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11x.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(
    data="/data/ephemeral/home/MCG/yolo_dataset_split/data.yaml",
    epochs=500,
    imgsz=2048,
    batch=3,
    hsv_h=0.0,         # Hue shift 비활성화
    hsv_s=0.0,         # Saturation shift 비활성화
    hsv_v=0.2,         # Brightness shift 비활성화
    degrees=0.2,       # 이미지 회전 비활성화
    translate=0.0,     # 이미지 이동 비활성화
    scale=0.2,   
    shear=0.2,         # 이미지 비틀기 비활성화
    perspective=0.0,   # 원근 변환 비활성화
    flipud=0.0,        # 상하 뒤집기 비활성화
    mosaic=0.0,        # Mosaic 비활성화
    mixup=0.0,         # Mixup 비활성화
    copy_paste=0.0 ,    # Copy-Paste 비활성화
    erasing=0.0,
    crop_fraction=0.0
)
#유지한 증강: Scale, degrees, shear, hsv_v, fliplr
