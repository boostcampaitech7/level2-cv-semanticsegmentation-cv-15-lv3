import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import numpy as np
import torch.optim as optim
from dataset import XRayDataset
from model import UNetPlusPlus
from train import train
import albumentations as A
from config import (
    BATCH_SIZE,
    CLASSES,
    LR,
    NUM_EPOCHS,
    IMAGE_ROOT,
    LABEL_ROOT,
    RANDOM_SEED
)
from utils import set_seed
def main():
    # Set random seed
    set_seed(RANDOM_SEED)
    
    # Print current working directory and image root for debugging
    print(f"IMAGE_ROOT: {IMAGE_ROOT}")
    print(f"LABEL_ROOT: {LABEL_ROOT}")
    
    # Walk through directories
    pngs = {
      os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
      for root, _dirs, files in os.walk(IMAGE_ROOT)
      for fname in files
      if os.path.splitext(fname)[1].lower() == ".png"
    }
                
    jsons = {
      os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
      for root, _dirs, files in os.walk(LABEL_ROOT)
      for fname in files
      if os.path.splitext(fname)[1].lower() == ".json"
    }
    
    if not pngs:
        raise ValueError(f"No PNG files found in {IMAGE_ROOT}")
    if not jsons:
        raise ValueError(f"No JSON files found in {LABEL_ROOT}")
    
    # Sort files
    pngs = sorted(pngs)
    jsons = sorted(jsons)
    print(f"Number of PNG files found: {len(pngs)}")
    print(f"Number of JSON files found: {len(jsons)}")
    pngs = np.array(pngs)
    jsons = np.array(jsons)
    # Split dataset using GroupKFold
    groups = [os.path.dirname(fname) for fname in pngs]
    ys = [0 for fname in pngs]
    gkf = GroupKFold(n_splits=5)

    train_filenames = []
    train_labelnames = []
    valid_filenames = []
    valid_labelnames = []
    
    for i, (x, y) in enumerate(gkf.split(pngs, ys, groups)):
        # 0번을 validation dataset으로 사용합니다.
        if i == 0:
            valid_filenames += list(pngs[y])
            valid_labelnames += list(jsons[y])

        else:
            train_filenames += list(pngs[y])
            train_labelnames += list(jsons[y])
    
    tf = A.Resize(512, 512)
    train_dataset = XRayDataset(train_filenames, train_labelnames, transforms=tf, is_train=True)
    val_dataset = XRayDataset(valid_filenames, valid_labelnames, transforms=tf, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    # Initialize model, criterion, and optimizer
    model = UNetPlusPlus(out_ch=len(CLASSES), supervision=False)
    # Loss function 정의
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer 정의
    optimizer = optim.RMSprop(params=model.parameters(), lr=LR, weight_decay=1e-6)
    
    # Train
    train(model, train_loader, val_loader, criterion, optimizer)

if __name__ == '__main__':
    main()