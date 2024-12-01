import os
import csv
import numpy as np
from ultralytics import YOLO
import json
import torch
from torch.utils.data import Dataset
from pycocotools import mask as mask_util

# RLE 인코딩 함수
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE 디코딩 함수
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)

# 모델 로드
model = YOLO("/data/ephemeral/home/jiwan/level2-cv-semanticsegmentation-cv-15-lv3/yolo/runs/segment/train/weights/best.pt")

# 예측할 이미지 폴더 경로 설정
image_folder = "/data/ephemeral/home/dataset_yolo/test"

# CSV 파일 생성 및 헤더 작성
csv_file_path = "/data/ephemeral/home/jiwan/level2-cv-semanticsegmentation-cv-15-lv3/yolo/result/predictions.csv"
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['image_name', 'class', 'rle'])

# 이미지 폴더 내 모든 이미지 파일 리스트 가져오기 (이미지 이름 순서 유지)
image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))])

# 클래스 이름 정렬 (미리 정의된 클래스 이름 리스트 사용 후 정렬)
class_names = [
    "finger-1", "finger-2", "finger-3", "finger-4", "finger-5",
    "finger-6", "finger-7", "finger-8", "finger-9", "finger-10",
    "finger-11", "finger-12", "finger-13", "finger-14", "finger-15",
    "finger-16", "finger-17", "finger-18", "finger-19",
    "Trapezium", "Trapezoid", "Capitate", "Hamate",
    "Scaphoid", "Lunate", "Pisiform", "Triquetrum",
    "Radius", "Ulna"
]

# 각 이미지에 대해 예측 수행 후 저장
for image_path in image_files:
    # 예측 수행
    results = model(image_path, imgsz = 2048)

    # 각 결과에 대해 처리
    class_rle_mapping = {class_name: '' for class_name in class_names}
    for result in results:
        # 마스크 결과 가져오기
        if result.masks is not None:
            # 박스와 마스크 데이터를 클래스 인덱스 순서대로 정렬
            sorted_results = sorted(zip(result.boxes, result.masks.data), key=lambda x: int(x[0].cls))

            for box, mask in sorted_results:
                # 클래스 가져오기
                class_idx = int(box.cls)
                class_name = class_names[class_idx]

                # 마스크를 numpy 배열로 변환하고 RLE로 인코딩
                mask_np = mask.cpu().numpy().astype(np.uint8)  # dtype을 uint8로 변환
                # 각 픽셀 값이 0 또는 1인지 확인 (이진화)
                mask_np = (mask_np > 0).astype(np.uint8)
                rle_str = encode_mask_to_rle(mask_np)  # 커스텀 RLE 인코딩 사용

                # 클래스에 해당하는 RLE 문자열 업데이트
                class_rle_mapping[class_name] = rle_str

    # 모든 클래스에 대해 결과 저장 (예측되지 않은 클래스는 빈 RLE 문자열 유지)
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for class_name, rle_str in class_rle_mapping.items():
            writer.writerow([os.path.basename(image_path), class_name, rle_str])

    print(f"Predictions for {image_path} logged in {csv_file_path}")
