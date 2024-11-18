import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from dataset.dataset import XRayInferenceDataset  # 데이터셋
from utils.utils import encode_mask_to_rle, decode_rle_to_mask, load_saved_model  # RLE 변환 함수, 결과 저장 함수
from config import SAVED_DIR, TRAIN_IMAGE_ROOT, VAL_BATCH_SIZE, CLASSES, IND2CLASS, ENCODER_NAME\
    , ENCODER_WEIGHTS
import segmentation_models_pytorch as smp
import albumentations as A

# 테스트 함수
def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()  # 이미지 텐서를 GPU로 이동
            outputs = model(images)

            # 결과 이미지 크기 맞추기
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)  # 예측값을 sigmoid 함수에 통과시켜 확률값으로 변환
            outputs = (outputs > thr).detach().cpu().numpy()  # 임계값(thr)을 기준으로 마스크 생성

            # RLE 인코딩 및 클래스별 파일 이름 저장
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class

# 메인 함수
def main():
    tf = A.Resize(512, 512)
    test_dataset = XRayInferenceDataset(transforms=tf)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=ENCODER_WEIGHTS,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=29,                     # model output channels (number of classes in your dataset)
    )
    load_saved_model(model,"./checkpoints/epoch_50_dice_0.9377.pt")

    rles, filename_and_class = test(model, test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
    })

    df.to_csv("./unetPlusPlus_effi4.csv", index=False)

if __name__ == "__main__":
    main()