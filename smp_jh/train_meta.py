import os
import time
import datetime
import random
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from datetime import timedelta
from config.config import Config
from dataset.dataset import XRayDataset, StratifiedXRayDataset
from models.model import get_model
from utils.metrics import dice_coef
from dataset.transforms import Transforms
from models.loss import Loss
from models.scheduler import Scheduler
from utils.wandb import init_wandb

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
    print(f"\nValidation Epoch {epoch}")
    val_start = time.time()
    model.eval()
    
    total_loss = 0
    total_focal_loss = 0
    total_msssim_loss = 0
    total_iou_loss = 0

    dices = []
    
    with torch.no_grad():
        for step, (images, masks) in enumerate(data_loader):
            # 데이터 타입 검증
            if not (isinstance(images, torch.Tensor) and isinstance(masks, torch.Tensor)):
                raise ValueError(
                    f"Expected images and masks to be torch.Tensor, but got images: {type(images)}, masks: {type(masks)}"
                )
        
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)
        
            # Forward pass
            outputs = model(images)
            
            # Resize outputs if needed
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
        
            # Calculate loss
            loss, focal_loss, ms_ssim_loss, iou_loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_focal_loss += focal_loss.item()
            total_msssim_loss += ms_ssim_loss.item()
            total_iou_loss += iou_loss.item()
        
            # Calculate dice score
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > threshold)
            dice = dice_coef(outputs, masks)
            dices.append(dice.detach().cpu())
            
            # 진행상황 출력 (10 step마다)
            if (step + 1) % 10 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Hybrid Loss: {round(loss.item(),4)}, '
                    f'focal Loss: {round(focal_loss.item(),4)}, '
                    f'msssim Loss: {round(ms_ssim_loss.item(),4)}, '
                    f'iou Loss: {round(iou_loss.item(),4)}, '
                    f'Dice: {round(torch.mean(dice).item(),4)}'
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
    avg_focal_loss = total_focal_loss / len(data_loader)
    avg_msssim_loss = total_msssim_loss / len(data_loader)
    avg_iou_loss = total_iou_loss / len(data_loader)
    
    # Print summary
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Validation hybrid Loss: {avg_loss:.4f}")
    print(f"Validation focal Loss: {avg_focal_loss:.4f}")
    print(f"Validation msssim Loss: {avg_msssim_loss:.4f}")
    print(f"Validation iou Loss: {avg_iou_loss:.4f}")
    print(f"Validation Time: {datetime.timedelta(seconds=val_time)}\n")
    
    # Create class-wise dice dictionary
    class_dice_dict = {
        f"{c}'s dice score": d.item() 
        for c, d in zip(Config.CLASSES, dices_per_class)
    }
    
    return avg_dice, class_dice_dict, avg_loss

def train():
    set_seed(Config.RANDOM_SEED)
    
    # Wandb 초기화
    init_wandb()

    # Device 설정
    cuda_number = 1
    torch.cuda.set_device(cuda_number)
    device = torch.device(f'cuda:{cuda_number}' if torch.cuda.is_available() else 'cpu')
    print(f"is_available cuda : {torch.cuda.is_available()}")
    print(f"current use : cuda({torch.cuda.current_device()})\n")
    
    # Mixed precision training
    scaler = GradScaler()

    # 모델 준비
    model = get_model(num_classes=len(Config.CLASSES)).to(device)

    criterion = Loss.get_criterion(Config.LOSS_TYPE)
    optimizer = optim.AdamW(
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
    # train_dataset = XRayDataset(
    #     image_root=Config.TRAIN_IMAGE_ROOT,
    #     label_root=Config.TRAIN_LABEL_ROOT,
    #     is_train=True,
    #     transforms=Transforms.get_train_transform()
    # )
    
    # valid_dataset = XRayDataset(
    #     image_root=Config.TRAIN_IMAGE_ROOT,
    #     label_root=Config.TRAIN_LABEL_ROOT,
    #     is_train=False,
    #     transforms=Transforms.get_valid_transform()
    # )

    train_dataset = StratifiedXRayDataset(
        image_root=Config.TRAIN_IMAGE_ROOT,
        label_root=Config.TRAIN_LABEL_ROOT,
        is_train=True,
        transforms=Transforms.get_train_transform(),
        meta_path=Config.META_PATH  # config에 META_PATH 추가 필요
    )
    
    valid_dataset = StratifiedXRayDataset(
        image_root=Config.TRAIN_IMAGE_ROOT,
        label_root=Config.TRAIN_LABEL_ROOT,
        is_train=False,
        transforms=Transforms.get_valid_transform(),
        meta_path=Config.META_PATH
    )

    # 데이터셋 통계 출력 (콘솔)
    print("\nDataset Statistics:")
    train_dataset.print_dataset_stats()
    valid_dataset.print_dataset_stats()
    
    # DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    # train_dataset.save_file_lists(output_dir="output_lists")
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    # valid_dataset.save_file_lists(output_dir="output_lists")
    
    # Training loop
    patience = 10 # 조기 종료 횟수
    counter = 0 # 조기 종료 카운터
    best_dice = 0.
    global_step = 0  # 전역 step 카운터 추가

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        epoch_focal_loss = 0
        epoch_msssim_loss = 0
        epoch_iou_loss = 0

        for step, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            # print(images.shape)
            # assert False
            masks = masks.to(device)
            
            with autocast(enabled=True):
                outputs = model(images)
                
                loss, focal_loss, ms_ssim_loss, iou_loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_focal_loss += focal_loss.item()
            epoch_msssim_loss += ms_ssim_loss.item()
            epoch_iou_loss += iou_loss.item()

            
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'hybrid Loss: {round(loss.item(),4)} '
                    f'focal Loss: {round(focal_loss.item(),4)} '
                    f'ms-ssim Loss: {round(ms_ssim_loss.item(),4)} '
                    f'iou Loss: {round(iou_loss.item(),4)} '
                )
                wandb.log({
                    "Step Loss": loss.item(),
                    "Learning Rate": optimizer.param_groups[0]['lr']
                }, step=global_step)
                global_step += 1

        # 에포크 종료 시간 계산
        epoch_time = time.time() - epoch_start

        # 평균 loss 계산 및 출력
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_focal_loss = epoch_focal_loss / len(train_loader)
        avg_epoch_msssim_loss = epoch_msssim_loss / len(train_loader)
        avg_epoch_iou_loss = epoch_iou_loss / len(train_loader)
        print("Epoch {}, Train Loss: {:.4f} || Elapsed time: {} || ETA: {}\n".format(
            epoch + 1,
            avg_epoch_loss,
            timedelta(seconds=epoch_time),
            timedelta(seconds=epoch_time * (Config.NUM_EPOCHS - epoch - 1))
        ))
        
        # Epoch 단위 로깅 - step을 epoch * len(train_loader)로 설정
        # Epoch 단위 로깅
        wandb.log({
            "Epoch": epoch + 1,
            "Train Hybrid Loss": avg_epoch_loss,
            "Train focal Loss": avg_epoch_focal_loss,
            "Train msssim Loss": avg_epoch_msssim_loss,
            "Train iou Loss": avg_epoch_iou_loss,
        }, step=global_step)
        
        if (epoch + 1) % Config.VAL_EVERY == 0:
            dice, dice_dict, val_loss = validation(
                epoch + 1, model, valid_loader, criterion, device, threshold=0.5
            )

            # # Scheduler step 추가
            # if Config.SCHEDULER_TYPE == "reduce":
            #     scheduler.step(dice)  # ReduceLROnPlateau는 metric을 전달
            # else:
            #     scheduler.step()      # 다른 스케줄러들은 단순히 step
            
            # Validation 결과 로깅
            wandb.log({
                "Validation Loss": val_loss,
                "Average Dice Score": dice,
                **dice_dict,
            }, step=global_step)

            # Best model 저장 및 Early Stopping 체크
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {Config.SAVED_DIR}")
                best_dice = dice
                torch.save(model, os.path.join(Config.SAVED_DIR, "best_model.pt"))
                
                # Best 모델 정보 로깅
                wandb.log({
                    "Best Dice Score": dice,
                    "Best Model Epoch": epoch + 1
                }, step=global_step)
                
                # Best Dice가 갱신되면 카운터 초기화
                counter = 0
            else:
                counter += 1
                print(f"Early Stopping counter: {counter} out of {patience}")
                
                if counter >= patience:
                    print(f"Early Stopping triggered! Best dice: {best_dice:.4f}")
                    wandb.log({
                        "Early Stopping": epoch + 1,
                        "Final Best Dice": best_dice
                    }, step=global_step)
                    break
    
    wandb.finish()

if __name__ == "__main__":
    train()