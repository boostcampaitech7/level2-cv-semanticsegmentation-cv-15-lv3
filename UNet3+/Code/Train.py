import datetime
from config import CLASSES, NUM_EPOCHS, VAL_EVERY, SAVED_DIR, MODELNAME,ACCUMULATION_STEPS
from Validation import validation
import os
import torch
from torch.cuda.amp import GradScaler, autocast
from Util.SetSeed import set_seed
from Util.DiscordAlam import send_discord_message

set_seed()

def save_model(model, file_name=MODELNAME):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

def train(model, data_loader, val_loader, criterion, optimizer, scheduler, accumulation_steps=ACCUMULATION_STEPS, threshold=0.92):
    """
    Args:
        accumulation_steps (int): Number of steps to accumulate gradients before updating.
        threshold (float): Dice 점수를 기준으로 손실 함수 변경.
    """
    print(f'Start training with Gradient Accumulation (accumulation_steps={accumulation_steps})..')
    model.cuda()

    n_class = len(CLASSES)
    best_dice = 0.0

    # Mixed Precision Scaler 생성
    scaler = GradScaler()

    for epoch in range(NUM_EPOCHS):
        # 에폭 시작 시간 기록
        start_time = datetime.datetime.now()
        print(f"Epoch {epoch + 1} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        model.train()

        # Gradient Accumulation Step 초기화
        optimizer.zero_grad()

        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()

            with autocast():  # Mixed Precision 모드
                outputs = model(images)  # 모델 출력 (d1, d2, d3, d4, d5)
                weights = [0.43, 0.35, 0.2, 0.15, 0.1]  # 가중치 설정

                loss = 0
                focal, ms_ssim_loss, iou, dice = 0, 0, 0, 0

                for i, output in enumerate(outputs):  # Deep Supervision 출력별 손실 계산
                    batch_loss, batch_focal, batch_ms_ssim, batch_iou, batch_dice = criterion(output, masks)
                    weighted_loss = weights[i] * batch_loss
                    loss += weighted_loss

                    # 개별 손실 성분 계산
                    focal += weights[i] * batch_focal
                    ms_ssim_loss += weights[i] * batch_ms_ssim
                    iou += weights[i] * batch_iou
                    dice += weights[i] * batch_dice

            # Mixed Precision Loss Scaling 및 Backpropagation
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (step + 1) % 80 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(), 4)} | '
                    f'Focal: {round(focal.item(), 4)}, '
                    f'MS-SSIM: {round(ms_ssim_loss.item(), 4)}, '
                    f'IoU: {round(iou.item(), 4)}, '
                    f'Dice: {round(dice.item(), 4)}'
                )


        # 마지막 미니배치 처리 후 Gradient 업데이트
        if (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Validation 주기에 따른 Loss 출력 및 Best Model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader)

            # Validation 결과에 따른 손실 함수 선택
            if dice < threshold:
                print(f"Validation Dice ({dice:.4f}) < Threshold ({threshold}), using IoU Loss.")
                criterion.delta = 0  # Dice Loss 비활성화
                criterion.beta = 1   # IoU Loss 활성화
            else:
                print(f"Validation Dice ({dice:.4f}) >= Threshold ({threshold}), using Dice Loss.")
                criterion.delta = 1  # Dice Loss 활성화
                criterion.beta = 0   # IoU Loss 비활성화

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                send_discord_message(f"성능 모니터링: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)

        # 스케줄러 업데이트
        scheduler.step()
        print(f"Epoch {epoch + 1}: Learning Rate -> {scheduler.get_last_lr()}")

        # 에폭 종료 시간 기록
        end_time = datetime.datetime.now()
        print(f"Epoch {epoch + 1} ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Epoch {epoch + 1} duration: {str(end_time - start_time)}")

        epoch_duration = end_time - start_time
        # 첫 에폭 시간 저장 및 ETA 계산
        if epoch == 0:
            first_epoch_time = epoch_duration
            estimated_total_time = first_epoch_time * NUM_EPOCHS
            eta = start_time + estimated_total_time
            send_discord_message(
                f"첫 번째 에폭의 소요 시간: {str(epoch_duration)}\n"
                f"전체 학습이 완료될 것으로 예상되는 시간: {eta.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"예상되는 총 남은 시간: {str(estimated_total_time)}"
            )
