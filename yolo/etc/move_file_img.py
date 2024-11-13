import os
import shutil
from glob import glob
from natsort import natsorted  # 자연스러운 정렬을 위한 라이브러리

# 원본 이미지가 있는 폴더 경로와 새로 만들 train/val 폴더 경로를 지정
base_folder = '/data/ephemeral/home/dataset_yolo/train/images'  # 원본 이미지 폴더 경로
train_folder = '/data/ephemeral/home/dataset_yolo/train/images/train'     # train 폴더 경로
val_folder = '/data/ephemeral/home/dataset_yolo/train/images/val'         # val 폴더 경로

# train과 val 폴더 생성
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# 모든 이미지 파일 가져오기 (jpg, png 등 확장자에 따라 조정 가능)
image_files = natsorted(glob(os.path.join(base_folder, '*.*')))  # 자연 정렬

# 80% train, 20% val로 분할
split_index = int(len(image_files) * 0.8)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# train 폴더로 이동
for file_path in train_files:
    shutil.move(file_path, os.path.join(train_folder, os.path.basename(file_path)))

# val 폴더로 이동
for file_path in val_files:
    shutil.move(file_path, os.path.join(val_folder, os.path.basename(file_path)))

print(f"Train 이미지 개수: {len(train_files)}, Val 이미지 개수: {len(val_files)}")
