import os
from sklearn.model_selection import GroupKFold
import albumentations as A
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import importlib

import config
from DataSet.DataLoder import get_image_label_paths
from DataSet.Dataset import XRayDataset
from Loss.Loss import CombinedLoss
from Util.DiscordAlam import send_discord_message
from TrainTool.MaskRpeatTrain import train
from Util.SetSeed import set_seed


def get_model_class(model_name):
    module_name, class_name = model_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def main():
    set_seed()

    if not os.path.isdir(config.SAVED_DIR):
        os.makedirs(config.SAVED_DIR)

    pngs, jsons = get_image_label_paths(IMAGE_ROOT=config.IMAGE_ROOT, LABEL_ROOT=config.LABEL_ROOT)

    # GroupKFold split
    groups = [os.path.dirname(fname) for fname in pngs]
    ys = [0 for fname in pngs]
    gkf = GroupKFold(n_splits=5)

    train_filenames, train_labelnames = [], []
    valid_filenames, valid_labelnames = [], []

    for i, (x, y) in enumerate(gkf.split(pngs, ys, groups)):
        if i == 0:
            valid_filenames += list(pngs[y])
            valid_labelnames += list(jsons[y])
        else:
            train_filenames += list(pngs[y])
            train_labelnames += list(jsons[y])

    train_dataset = XRayDataset(
        train_filenames,
        train_labelnames,
        transforms=None,
        is_train=True,
    )
    valid_dataset = XRayDataset(
        valid_filenames,
        valid_labelnames,
        transforms=None,
        is_train=False,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    # 동적으로 모델 클래스 로드
    model_name = f"Model.{config.MODEL_NAME}"
    ModelClass = get_model_class(model_name)
    model = ModelClass(n_classes=len(config.CLASSES))

    # Loss function 정의
    criterion = CombinedLoss(
        focal_weight=1,
        iou_weight=1,
        ms_ssim_weight=1,
        dice_weight=0,
        boundary_weight=1
    )

    # Optimizer 및 Scheduler 정의
    optimizer = optim.AdamW(params=model.parameters(), lr=config.LR, weight_decay=2e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.4, patience=5, verbose=True
    )

    send_discord_message(f"# 실험: {config.EXPERIMENT_NAME}")
    train(model, train_loader, valid_loader, criterion, optimizer, scheduler)


if __name__ == "__main__":
    main()
