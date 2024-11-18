import os
import time
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
from models.loss import Loss
from models.scheduler import Scheduler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)     

def validation(epoch, model, data_loader, criterion, device, threshold=0.5):
    """
    Validation function with improved monitoring and GPU utilization
    
    Args:
        epoch (int): Current epoch number
        model (nn.Module): Model to validate
        data_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss criterion
        device (torch.device): Device to use
        threshold (float): Threshold for binary prediction
    
    Returns:
        tuple: (average_dice, class_dice_dict, average_loss)
    """
    val_start = time.time()
    model.eval()
    
    total_loss = 0
    dices = []
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc=f'[Validation Epoch {epoch}]') as pbar:
            for step, (images, masks) in enumerate(data_loader):
                # Move data to GPU
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Resize outputs if needed
                output_h, output_w = outputs.size(-2), outputs.size(-1)
                mask_h, mask_w = masks.size(-2), masks.size(-1)
                if output_h != mask_h or output_w != mask_w:
                    outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                
                # Calculate loss
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                
                # Calculate dice score on GPU
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > threshold)
                dice = dice_coef(outputs, masks)
                dices.append(dice.detach().cpu())
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(
                    dice=torch.mean(dice).item(),
                    loss=loss.item()
                )
    
    # Calculate validation time
    val_time = time.time() - val_start
    
    # Calculate final metrics
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    
    # Create formatted dice score string
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(Config.CLASSES, dices_per_class)
    ]
    print("\n".join(dice_str))
    
    # Calculate average metrics
    avg_dice = torch.mean(dices_per_class).item()
    avg_loss = total_loss / len(data_loader)
    
    # Print summary
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Time: {datetime.timedelta(seconds=val_time)}\n")
    
    # Create class-wise dice dictionary
    class_dice_dict = {
        f"{c}'s dice score": d.item() 
        for c, d in zip(Config.CLASSES, dices_per_class)
    }
    
    return avg_dice, class_dice_dict, avg_loss

def train():
    set_seed(Config.RANDOM_SEED)
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 준비
    model = get_model(num_classes=len(Config.CLASSES)).to(device)
    criterion = Loss.get_criterion(Config.LOSS_TYPE)
    optimizer = optim.Adam(
        params=model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=1e-6
    )
    scheduler = Scheduler.get_scheduler(
        scheduler_type=Config.SCHEDULER_TYPE,
        optimizer=optimizer,
        min_lr=Config.MIN_LR
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
    # Training loop
    best_dice = 0.
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        
        for step, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
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
            dice, dice_dict, val_loss = validation(
                epoch + 1, 
                model, 
                valid_loader, 
                criterion,
                device,
                threshold=0.5
            )

            # Scheduler step 추가
            # if Config.SCHEDULER_TYPE == "reduce":
            #     scheduler.step(dice)  # ReduceLROnPlateau는 metric을 전달
            # else:
            #     scheduler.step()      # 다른 스케줄러들은 단순히 step
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {Config.SAVED_DIR}")
                best_dice = dice
                torch.save(model, os.path.join(Config.SAVED_DIR, "best_model.pt"))

if __name__ == "__main__":
    train()