import os
import json
import random
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import albumentations as A

from dataset import XRayInferenceDataset
from utils import encode_mask_to_rle, decode_rle_to_mask
from smp_model import get_smp_model
from config import CLASSES, IND2CLASS, MODEL_NAME

warnings.filterwarnings('ignore')

def load_model(model_path):
    """학습된 모델을 불러오는 함수"""
    # 동일한 모델 아키텍처 생성
    model = get_smp_model(
        model_type="unetplusplus",
        encoder_name="resnet50",
        encoder_weights=None
    )
    
    # 학습된 가중치 로드
    checkpoint = torch.load(model_path)
    
    # 체크포인트 구조에 따른 다양한 로딩 방식 처리
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.cuda()
    return model

def test(model_path: str = '../checkpoints/UNetPlusPlus_simple/best_model.pt') -> None:
    """
    테스트 데이터에 대한 추론을 실행하고 결과를 CSV 파일로 저장
    Args:
        model_path: 저장된 모델 체크포인트 경로
    """
    # transform 정의
    tf = A.Compose([
        A.Normalize()
    ])
    
    # 데이터셋 및 데이터로더 설정
    test_dataset = XRayInferenceDataset(transforms=tf)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    # 모델 불러오기
    model = load_model(model_path)
    
    # 추론 실행
    rles, filename_and_class = inference(model, test_loader)
    
    # submission 파일 생성
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    df.to_csv(f"{MODEL_NAME}_output.csv", index=False)

def inference(model, data_loader, thr=0.5):
    """데이터로더에 대한 추론 실행"""
    model.eval()
    
    rles = []
    filename_and_class = []
    
    with torch.no_grad():
        for imgs, image_names in tqdm(data_loader):
            imgs = imgs.cuda()
            
            # 모델 추론
            masks = model(imgs)
            masks = torch.sigmoid(masks)
            masks = (masks > thr).detach().cpu().numpy()
            
            # batch 내 각 이미지에 대해 처리
            for i, mask in enumerate(masks):
                for c, segm in enumerate(mask):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_names[i]}")
    
    return rles, filename_and_class

if __name__ == "__main__":
    test()

