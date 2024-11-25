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
        for step, (images, image_names, crop_boxes) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            outputs = model(images)

            for output, image_name, crop_box in zip(outputs, image_names, crop_boxes):
                print(crop_box)
                start_x, start_y, end_x, end_y = crop_box
                crop_width = end_x - start_x
                crop_height = end_y - start_y
                outputs = torch.sigmoid(outputs)
                # Interpolate to crop size
                output = F.interpolate(output.unsqueeze(0), size=(crop_height, crop_width), mode="bilinear")
                output = (output > thr).squeeze(0).detach().cpu().numpy()

                for c, segm in enumerate(output):
                    full_size_mask = np.zeros((2048, 2048), dtype=np.uint8)
                    full_size_mask[start_y:end_y, start_x:end_x] = segm
                    rle = encode_mask_to_rle(full_size_mask)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # CUDA 문제 해결을 위한 spawn 방식 설정

    # 모델 로드
    model = torch.load(os.path.join(SAVED_DIR, INFERENCE_MODEL_NAME))
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
        batch_size=2,
        shuffle=False,
        num_workers=0,  # 멀티프로세싱 비활성화
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    # 테스트 수행
    rles, filename_and_class = test(model, test_loader)

    # 결과 저장
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(os.path.join(CSVDIR, CSVNAME), index=False)
