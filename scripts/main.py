import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import numpy as np

from dataset import XRayDataset
from model import UNetPlusPlus
from train import train
from config import (
    BATCH_SIZE,
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
    
    # Get file paths
    pngs = []
    jsons = []
    
    # Print current working directory and image root for debugging
    print(f"Current working directory: {os.getcwd()}")
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
    
    # Split dataset using GroupKFold
    groups = [os.path.dirname(fname) for fname in pngs]
    gkf = GroupKFold(n_splits=5)
    
    # Get train/val indices for first fold
    train_idx, val_idx = next(gkf.split(pngs, groups=groups))
    
    # Create datasets
    train_dataset = XRayDataset(
        np.array(pngs)[train_idx].tolist(),
        np.array(jsons)[train_idx].tolist(),
        transforms=None,  # Add your transforms here
        is_train=True
    )
    
    val_dataset = XRayDataset(
        np.array(pngs)[val_idx].tolist(),
        np.array(jsons)[val_idx].tolist(),
        transforms=None,  # Add your transforms here
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model, criterion, and optimizer
    model = UNetPlusPlus()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Train
    train(model, train_loader, val_loader, criterion, optimizer)

if __name__ == '__main__':
    main()