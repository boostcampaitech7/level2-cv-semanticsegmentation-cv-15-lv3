import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import numpy as np
import torch.optim as optim
from dataset import XRayDataset
from model import DUCKNet, UNetPlusPlus
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
from smp_model import get_smp_model

def main():
    # Set random seed
    set_seed(RANDOM_SEED)
    # Initialize model, criterion, and optimizer
    # model = UNetPlusPlus(out_ch=len(CLASSES), supervision=False,
                        #  height=HEIGHT, width=WIDTH)
    model = get_smp_model(
        model_type="unetplusplus",
        encoder_name="",
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(CLASSES)
    )
    model = model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(params=model.parameters(), lr=LR, weight_decay=1e-6)

    # Train
    MODE = "test"
    if MODE == "train":
        train(model, criterion, optimizer, discord_alert=DISCORD_ALERT)
    elif MODE == "test":
        test(model)

if __name__ == '__main__':
    main()