from ultralytics import YOLO
import torch

# 모델 설정 정보
yaml_path = "yolo11x-seg.yaml"  # 새로운 모델 설정을 위한 YAML 파일
pretrained_weights = "yolo11x-seg.pt"  # 사전 훈련된 가중치 파일 경로
transfer_weights = "yolo11x.pt"  # YAML 빌드 시 사용할 가중치 파일 경로

# 모델 빌드 또는 가중치 로드
model = YOLO(yaml_path)  # YAML 파일로 새 모델 빌드
model = YOLO(pretrained_weights)  # 사전 훈련된 모델 로드
model = YOLO(yaml_path).load(transfer_weights)  # YAML로 빌드하고 가중치 로드

# 모델 훈련 설정
data_path = "/data/ephemeral/home/jiwan/yolo/data.yaml"  # 데이터 설정 파일 경로
epochs = 250
imgsz = 1280
batch_size = 4  # 추가한 batch_size 설정

# GPU 메모리 정리
torch.cuda.empty_cache()

# 모델 훈련 시작
results = model.train(
    data=data_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size
)

print("훈련이 완료되었습니다.")
