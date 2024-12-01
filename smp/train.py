import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset.dataset import XRayDataset
import albumentations as A
from config import TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, SAVED_DIR, NUM_EPOCHS, \
    CLASSES, WEIGHT_DECAY, RANDOM_SEED, LR, ENCODER_NAME, NUM_WORKERS, ENCODER_WEIGHTS
from utils.utils import dice_coef, save_model, set_seed
from tqdm import tqdm
import datetime
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
import os

# 시드 설정 (재현 가능성 보장)
set_seed()

# WandB 초기화
wandb.init(project="hand_bone_segmentation", name="unetplusplus_effi4")

# WandB 하이퍼파라미터 설정
wandb.config.update({
    "train_batch_size": TRAIN_BATCH_SIZE,
    "val_batch_size": VAL_BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LR,
    "random_seed": RANDOM_SEED,
    "encoder_name": ENCODER_NAME,
    "encoder_weights": ENCODER_WEIGHTS,
    "weight_decay": WEIGHT_DECAY
})

# 데이터셋 로딩
tf = A.Resize(512, 512)
train_dataset = XRayDataset(is_train=True, transforms=tf)
valid_dataset = XRayDataset(is_train=False, transforms=tf)

# 훈련 및 검증 데이터 로더 설정
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True,
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False
)

# 모델 정의
model = smp.UnetPlusPlus(
    encoder_name=ENCODER_NAME,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=ENCODER_WEIGHTS,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=29,                      # model output channels (number of classes in your dataset)
).cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# 검증 함수 정의
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
            
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation
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
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    # WandB 로그 (이미지 제외)
    wandb.log({
        "epoch": epoch,
        "val_loss": total_loss / cnt,
        "avg_dice": avg_dice,
        **{f"dice_{c}": d.item() for c, d in zip(CLASSES, dices_per_class)}  # 클래스별 Dice
    }, step=epoch)

    return avg_dice

import time  # 추가

def train(model, train_loader, valid_loader, criterion, optimizer):
    best_models = []  # 상위 3개의 모델 저장
    patience = 10     # 조기 종료 기준
    no_improve_count = 0
    best_dice = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        # 에폭 시작 시간 기록
        epoch_start_time = time.time()

        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            outputs = model(images)

            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # WandB 로그 (에포크 단위 손실)
        wandb.log({
            "epoch": epoch,  # epoch 단위로 로그
            "train_loss": total_loss / len(train_loader)
        }, step=epoch)
        print(f'Epoch [{epoch}/{NUM_EPOCHS}], Loss: {total_loss / len(train_loader):.4f}')
        
        # 검증 및 모델 저장
        avg_dice = validation(epoch, model, valid_loader, criterion)

        # 에폭 종료 시간 기록
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # 남은 에폭 시간 예측
        remaining_epochs = NUM_EPOCHS - epoch
        estimated_time_remaining = epoch_duration * remaining_epochs
        print(f"Epoch {epoch} duration: {epoch_duration:.2f} seconds")
        print(f"Estimated time remaining: {estimated_time_remaining / 60:.2f} minutes")
        
        if epoch >= 20:  # 20 에폭부터 조기 종료 카운트 시작
            if avg_dice <= best_dice:
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} epoch(s).")
                if no_improve_count >= patience:
                    print(f"Early stopping triggered after {patience} epochs.")
                    break
            else:
                no_improve_count, best_dice = 0, avg_dice
                print(f"New best Dice at epoch {epoch}: {avg_dice:.4f}")
                save_path = f"{SAVED_DIR}/epoch_{epoch}_dice_{best_dice:.4f}.pt"
                save_model(model, save_path)
                best_models.append((avg_dice, save_path))
                best_models.sort(key=lambda x: x[0], reverse=True)

                if len(best_models) > 3:
                    _, to_delete = best_models.pop(-1)
                    os.remove(to_delete)
                    print(f"Removed lowest Dice model: {to_delete}")
        else:
            print(f"Skipping early stopping until epoch 20. Current epoch: {epoch}")


if __name__ == '__main__':
    train(model, train_loader, valid_loader, criterion, optimizer)