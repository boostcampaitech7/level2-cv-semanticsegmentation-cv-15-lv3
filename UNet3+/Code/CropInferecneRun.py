import albumentations as A
from torch.utils.data import DataLoader, Dataset
from config import IND2CLASS, SAVED_DIR, INFERENCE_MODEL_NAME, IMSIZE, CSVDIR, CSVNAME, CLASSES, TEST_IMAGE_ROOT, SAVE_VISUALIZE_TRAIN_DATA_PATH
import os
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm.auto import tqdm
from ultralytics import YOLO
from DataSet.YoloInferenceDataset import XRayInferenceDataset
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np


def keep_largest_connected_component(mask):
    """
    입력된 이진 마스크에서 가장 큰 연결 요소만 남기고 나머지를 제거.
    :param mask: 이진 마스크 (H, W)
    :return: 가장 큰 연결 요소만 남은 이진 마스크 (H, W)
    """
    # 연결 요소 분석
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

    # 연결 요소 중 가장 큰 영역 찾기 (배경 제외: stats[1:] 사용)
    if num_labels <= 1:  # 연결 요소가 없으면 원래 마스크 반환
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 가장 큰 영역의 인덱스 (1부터 시작)
    
    # 가장 큰 연결 요소만 남기기
    largest_component = (labels == largest_label).astype(np.uint8)
    return largest_component


def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        # 클래스별 색상 정의
        CLASS_COLORS = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 128, 128),# Gray
            (255, 165, 0)   # Orange
        ]

        for step, (images, image_names, crop_boxes) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            print("!!!!!",images.shape)
            outputs = model(images)
            #outputs = torch.sigmoid(outputs)
            #print("()()", outputs.shape)
            
            for output, image, image_name,  crop_box in zip(outputs, images, image_names, crop_boxes):
                start_x, start_y, end_x, end_y = crop_box
                output = torch.sigmoid(output)
                crop_width = end_x - start_x
                crop_height = end_y - start_y

                # 출력 크기와 crop 크기 확인
                output_height, output_width = output.shape[-2:]
                if (output_width != crop_width) or (output_height != crop_height):
                    print(f"Output size: ({output_width}, {output_height}), "
                        f"Crop size: ({crop_width}, {crop_height})")
                    print(f"Image name: {image_name}, Step: {step}")
                    raise ValueError("Output size does not match crop size!")
                
                output = output.detach().cpu().numpy()  # Tensor -> Numpy
                image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # Normalize to 0-255
                
                # 각 클래스에 대해 시각화
                overlay_mask = np.zeros_like(image, dtype=np.uint8)  # Overlay 초기화
                for c, segm in enumerate(output):  # segm: (H, W)
                    binary_mask = (segm > thr).astype(np.uint8)
                    #largest_component = keep_largest_connected_component(binary_mask)

                    # Crop된 영역을 원본 크기로 변환
                    #full_size_mask = np.zeros((2048, 2048), dtype=np.uint8)
                    #full_size_mask[start_y:end_y, start_x:end_x] = largest_component

                    # 마스크를 색상으로 변환
                    color = CLASS_COLORS[c]
                    color_mask = np.zeros_like(overlay_mask, dtype=np.uint8)
                    for i in range(3):  # RGB 채널별로 색상 적용
                        color_mask[:, :, i] = binary_mask * color[i]

                    # Overlay에 누적
                    overlay_mask = cv2.add(overlay_mask, color_mask)

                # 원본 이미지와 마스크 합성
                blended_image = cv2.addWeighted(image, 0.7, overlay_mask, 0.3, 0)
                clean_image_name = os.path.basename(image_name)
                # 이미지 저장last_last_last
                save_path = f"/data/ephemeral/home/MCG/YOLO_Detection_Model/test_last/{clean_image_name}"
                print(save_path,blended_image.shape)
                cv2.imwrite(save_path, cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
                print(f"Visualization saved at {save_path}")

                # RLE 생성 및 추가 (기존 로직 유지)
                for c, segm in enumerate(output):
                    binary_mask = (segm > thr).astype(np.uint8)
                    largest_component = keep_largest_connected_component(binary_mask)
                    full_size_mask = np.zeros((2048, 2048), dtype=np.uint8)
                    full_size_mask[start_y:end_y, start_x:end_x] = largest_component
                    rle = encode_mask_to_rle(full_size_mask)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")



    return rles, filename_and_class

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # CUDA 문제 해결을 위한 spawn 방식 설정

    # 모델 로드
    model = torch.load(os.path.join(SAVED_DIR, INFERENCE_MODEL_NAME))
    print(os.path.join(SAVED_DIR, INFERENCE_MODEL_NAME))
    yolo_model = YOLO("/data/ephemeral/home/MCG/YOLO_Detection_Model/best.pt") # YOLO 모델을 GPU로 이동

    # PNG 파일 가져오기
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=TEST_IMAGE_ROOT)
        for root, _dirs, files in os.walk(TEST_IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    # 데이터셋 생성
    test_dataset = XRayInferenceDataset(
        filenames=pngs,
        yolo_model=yolo_model,  # YOLO 모델 전달
        save_dir=SAVE_VISUALIZE_TRAIN_DATA_PATH,
        draw_enabled=True
    )
    def custom_collate_fn(batch):
        images, image_names, crop_boxes = zip(*batch)
        return (
            torch.stack(images),  # 이미지 텐서 병합
            list(image_names),    # 파일명 리스트 유지
            list(crop_boxes)      # crop_box 리스트 유지
        )
    
    # 데이터 로더 생성
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 멀티프로세싱 비활성화
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    # 테스트 수행
    rles, filename_and_class = test(model, test_loader,thr=0.5)

    # 결과 저장
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(os.path.join(CSVDIR, CSVNAME), index=False)
