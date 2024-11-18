import os
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config.config import Config
from dataset.dataset import XRayDataset
from models.model import get_model
from utils.metrics import dice_coef
from dataset.transforms import Transforms

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)     

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            outputs = model(images)['out']
            
            # Resize outputs if needed
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    
    # Print dice scores for each class
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(Config.CLASSES, dices_per_class)
    ]
    print("\n".join(dice_str))

    avg_dice = torch.mean(dices_per_class).item()
    return avg_dice

def train():
    set_seed(Config.RANDOM_SEED)
    
    # 모델 준비
    model = get_model(num_classes=len(Config.CLASSES))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        params=model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=1e-6
    )
    
    # 데이터셋 준비
    train_dataset = XRayDataset(
        image_root=Config.TRAIN_IMAGE_ROOT,
        label_root=Config.TRAIN_LABEL_ROOT,
        is_train=True,
        transforms=Transforms.get_train_transform()
    )
    
    valid_dataset = XRayDataset(
        image_root=Config.TRAIN_IMAGE_ROOT,
        label_root=Config.TRAIN_LABEL_ROOT,
        is_train=False,
        transforms=Transforms.get_valid_transform()
    )
    
    # DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    # Training loop
    best_dice = 0.
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
        
        if (epoch + 1) % Config.VAL_EVERY == 0:
            dice = validation(epoch + 1, model, valid_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {Config.SAVED_DIR}")
                best_dice = dice
                torch.save(model, os.path.join(Config.SAVED_DIR, "best_model.pt"))

if __name__ == "__main__":
    train()