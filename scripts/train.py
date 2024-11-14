import datetime
import os
import numpy as np
from sklearn.model_selection import GroupKFold
import torch
from tqdm import tqdm
from config import BATCH_SIZE, HEIGHT, IMAGE_ROOT, LABEL_ROOT, MODEL_NAME, NUM_EPOCHS, CLASSES, RANDOM_SEED, SAVED_DIR, SERVER_ID, VAL_EVERY, DISCORD_ALERT, WIDTH  # Add CLASSES to the import
from dataset import XRayDataset
from utils import save_model, dice_coef, send_discord_message, set_seed
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import albumentations as A

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.cuda()
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()

            outputs = model(images)

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # restore original size
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

    return avg_dice

def train(model, criterion, optimizer, discord_alert=DISCORD_ALERT):
    set_seed(RANDOM_SEED)
    server_id = SERVER_ID
    train_loader, val_loader = train_val_return()
    
    if discord_alert:
        send_discord_message(f"🎬 [서버 {server_id}] {MODEL_NAME} 학습이 시작되었습니다!")
    
    print(f'Start training..')
    model.cuda()

    n_class = len(CLASSES)
    best_dice = 0.

    for epoch in range(NUM_EPOCHS):
        model.train()
        
        # 10 에폭 시작시 현재 진행상황 알림
        if (epoch) % 10 == 0 and discord_alert:
            send_discord_message(f"📊 [서버 {server_id}] 현재 진행상황: Epoch [{epoch+1}/{NUM_EPOCHS}] 진행 중")

        for step, (images, masks) in enumerate(train_loader):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()

            # inference
            outputs = model(images)

            # loss 계산
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)

    # 학습 종료 알림
    if discord_alert:
        send_discord_message(f"✨ [서버 {server_id}] 학습이 완료되었습니다!\n"
                            f"최종 최고 성능: {best_dice:.4f}")

def train_val_return():
    # Print current working directory and image root for debugging
    print(f"IMAGE_ROOT: {IMAGE_ROOT}")
    print(f"LABEL_ROOT: {LABEL_ROOT}")
    
    # Get train/val files
    train_filenames, train_labelnames, valid_filenames, valid_labelnames = get_train_val_files(IMAGE_ROOT, LABEL_ROOT)
    
    tf = A.Resize(HEIGHT, WIDTH)
    train_dataset = XRayDataset(train_filenames, train_labelnames, transforms=tf, is_train=True)
    val_dataset = XRayDataset(valid_filenames, valid_labelnames, transforms=tf, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )
    
    return train_loader, val_loader

def get_train_val_files(image_root, label_root):
    # Walk through directories
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
                
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=label_root)
        for root, _dirs, files in os.walk(label_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
    
    if not pngs:
        raise ValueError(f"No PNG files found in {image_root}")
    if not jsons:
        raise ValueError(f"No JSON files found in {label_root}")
    
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
            
    return train_filenames, train_labelnames, valid_filenames, valid_labelnames