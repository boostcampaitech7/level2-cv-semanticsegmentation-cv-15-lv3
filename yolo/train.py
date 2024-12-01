from ultralytics import YOLO
import torch
import yaml

# 모델 설정 정보
yaml_path = "yolo11x-seg.yaml"  # 새로운 모델 설정을 위한 YAML 파일 경로
pretrained_weights = "yolo11x-seg.pt"  # 사전 학습된 가중치 파일 경로
transfer_weights = "yolo11x.pt"  # YAML로 모델을 생성할 때 사용할 가중치 파일 경로

# 증강기 설정 파일 불러오기
augmenter_config_path = "./args.yaml"  # 데이터 증강 설정을 위한 YAML 파일 경로
with open(augmenter_config_path, 'r') as file:
    augmenter_args = yaml.safe_load(file)  # args.yaml 파일을 읽어서 설정 정보를 로드

# 모델 생성 또는 가중치 로드
model = YOLO(yaml_path).load(transfer_weights)

# 모델 학습 설정
data_path = "/data/ephemeral/home/jiwan/level2-cv-semanticsegmentation-cv-15-lv3/yolo/data.yaml"  # 데이터 구성 파일 경로
epochs = 100  # 학습 반복 횟수
imgsz = 2048  # 입력 이미지 크기
batch_size = 1  # 배치 크기 설정

# GPU 메모리 정리
torch.cuda.empty_cache()

# 모델 학습 시작
# args.yaml에서 불러온 설정을 반영하고, 중복된 'data' 키를 제거한 상태로 학습을 진행
results = model.train(
    data=data_path,  # 데이터 경로 설정
    epochs=epochs,  # 학습 반복 횟수 설정
    imgsz=imgsz,  # 입력 이미지 크기 설정
    batch=batch_size,  # 배치 크기 설정
    augment=True,  # 데이터 증강을 사용하도록 설정
    #**augmenter_args  # args.yaml에서 로드한 증강 설정 추가 적용 (data는 제거된 상태)
)

print("Training completed.")  # 학습 완료 메시지 출력
