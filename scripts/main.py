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
from test import test
import albumentations as A
from config import (
    CLASSES,
    HEIGHT,
    LR,
    RANDOM_SEED,
    MODE,
    DISCORD_ALERT,
    WIDTH
)

from utils import set_seed
def main():
    # Set random seed
    set_seed(RANDOM_SEED)
    
    # Initialize model, criterion, and optimizer
    model = UNetPlusPlus(out_ch=len(CLASSES), supervision=False,
                         height=HEIGHT, width=WIDTH)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(params=model.parameters(), lr=LR, weight_decay=1e-6)

    # Train
    if MODE == "train":
        train(model, criterion, optimizer, discord_alert=DISCORD_ALERT)
    elif MODE == "test":
        test(model)

if __name__ == '__main__':
    main()