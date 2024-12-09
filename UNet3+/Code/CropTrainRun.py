import os
import importlib
from sklearn.model_selection import GroupKFold
import albumentations as A
# torch
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from DataSet.DataLoder import get_image_label_paths
from DataSet.LabelBaseCropDataset import XRayDataset
from Loss.Loss import CombinedLoss
from TrainTool.MaskRpeatTrain import train
from Util.SetSeed import set_seed
from sklearn.utils import shuffle
from Util.cusom_cosine_annal import CosineAnnealingWarmUpRestarts

# 동적으로 모델 클래스를 가져오는 함수
def get_model_class(model_name):
    module_name, class_name = model_name.rsplit(".", 1)  # 모듈명과 클래스명 분리
    module = importlib.import_module(module_name)  # 동적으로 모듈 import
    return getattr(module, class_name)  # 클래스 가져오기

def main():
    set_seed()

    pngs, jsons = get_image_label_paths(IMAGE_ROOT=config.IMAGE_ROOT, LABEL_ROOT=config.LABEL_ROOT)

    # GroupKFold를 사용한 데이터 분리
    groups = [os.path.dirname(fname) for fname in pngs]
    ys = [0 for fname in pngs]
    gkf = GroupKFold(n_splits=4)

    train_filenames, train_labelnames = [], []
    valid_filenames, valid_labelnames = [], []

    for i, (x, y) in enumerate(gkf.split(pngs, ys, groups)):
        if i == 1:
            valid_filenames += list(pngs[y])
            valid_labelnames += list(jsons[y])
        else:
            train_filenames += list(pngs[y])
            train_labelnames += list(jsons[y])

    # 데이터셋 생성
    train_dataset1 = XRayDataset(train_filenames, train_labelnames, is_train=True)
    tf = A.Compose([
        A.Rotate(limit=14, p=0.5),
        A.HorizontalFlip(p=1),
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            brightness_by_max=False,
            p=0.8
        ),
    ])
    train_dataset2 = XRayDataset(train_filenames, train_labelnames, is_train=True, transforms=tf)
    train_dataset = train_dataset1 + train_dataset2

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    valid_dataset = XRayDataset(valid_filenames, valid_labelnames, is_train=False)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    # 동적으로 모델 가져오기
    model_name = f"Model.{config.MODEL}"  # ex) Model.4Stage_HRnet_UNet3+_from_last_stage
    ModelClass = get_model_class(model_name)
    model = ModelClass(n_classes=len(config.CLASSES))

    # Loss function 정의
    criterion = CombinedLoss(focal_weight=1, iou_weight=1, ms_ssim_weight=1, dice_weight=0)

    # Optimizer 정의
    optimizer = optim.AdamW(params=model.parameters(), lr=config.LR, weight_decay=5e-5)
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, T_0=50, T_mult=2, eta_max=0.0003, T_up=8, gamma=0.5
    )

    # Training
    train(model, train_loader, valid_loader, criterion, optimizer, scheduler)

if __name__ == "__main__":
    main()
