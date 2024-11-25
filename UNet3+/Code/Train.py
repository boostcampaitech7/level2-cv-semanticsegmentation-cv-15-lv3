import datetime
from config import CLASSES, NUM_EPOCHS, VAL_EVERY, SAVED_DIR, MODELNAME, ACCUMULATION_STEPS
from Validation import validation
import os
import torch
import wandb
from Util.SetSeed import set_seed
from Util.DiscordAlam import send_discord_message

set_seed()

def save_model(model, file_name=MODELNAME):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

def train(model, data_loader, val_loader, criterion, optimizer, scheduler, accumulation_steps=ACCUMULATION_STEPS, threshold=0.93):
    """
    Args:
        accumulation_steps (int): Number of steps to accumulate gradients before updating.
        threshold (float): Dice 점수를 기준으로 손실 함수 변경.
    """
    print(f'Start training with Gradient Accumulation (accumulation_steps={accumulation_steps})..')
    model.cuda()

    n_class = len(CLASSES)
    best_dice = 0.0
    global_step = 0

    # Early stopping 설정
    patience = 10
    counter = 0

    for epoch in range(NUM_EPOCHS):
        # 에폭 시작 시간 기록
        start_time = datetime.datetime.now()
        print(f"Epoch {epoch + 1} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        model.train()

        # Gradient Accumulation Step 초기화
        optimizer.zero_grad()

        epoch_loss = 0
        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()

            # Mixed Precision 제거
            outputs = model(images)  # 모델 출력 (d1, d2, d3, d4, d5)
            
            loss = 0
            focal, iou, dice,msssim = 0, 0, 0,0

            #for i, output in enumerate(outputs):  # Deep Supervision 출력별 손실 계산
            batch_loss, batch_focal, batch_iou, batch_dice, batch_msssim = criterion(outputs, masks)
            # 각 배치별 손실을 출력

            weighted_loss = batch_loss
            loss += weighted_loss

            # 개별 손실 성분 계산
            focal = batch_focal
            msssim = batch_msssim
            iou =  batch_iou
            dice =  batch_dice

            epoch_loss += loss.item()

            # Backpropagation
            loss.backward()
            print_loss=loss.item()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % 80 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(print_loss, 4)} | '
                    f'Focal: {round(focal.item(), 4)}, '
                    f'Msssim: {round(msssim.item(), 4)}, '
                    f'IoU: {round(iou.item(), 4)}, '
                    f'Dice: {round(dice.item(), 4)}'
                )

                # Step 단위 로깅
                wandb.log({
                    "Step Loss": loss.item(),
                    "Step Focal Loss": focal.item(),
                    # "Step Msssim Loss": msssim.item(),
                    "Step IoU": iou.item(),
                    "Step Dice": dice.item(),
                    "Learning Rate": optimizer.param_groups[0]['lr']
                }, step=global_step)
                global_step += 1
        
        # 에폭 평균 loss 계산 및 로깅
        avg_epoch_loss = epoch_loss / len(data_loader)
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": avg_epoch_loss,
        }, step=global_step)

        # Validation 주기에 따른 Loss 출력 및 Best Model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)

            '''# Validation 결과에 따른 손실 함수 선택
            if dice < threshold:
                print(f"Validation Dice ({dice:.4f}) < Threshold ({threshold}), using IoU Loss.")
                criterion.dice_weight = 0  # Dice Loss 비활성화
                criterion.iou_weight = 1   # IoU Loss 활성화
            else:
                print(f"Validation Dice ({dice:.4f}) >= Threshold ({threshold}), using Dice Loss.")
                criterion.dice_weight = 1  # Dice Loss 활성화
                criterion.iou_weight = 0   # IoU Loss 비활성화'''

            # Validation 결과 로깅
            wandb.log({
                "Validation Dice Score": dice,
            }, step=global_step)

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                send_discord_message(f"성능 모니터링: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)

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
                    wandb.finish()
                    break

        # 스케줄러 업데이트
        scheduler.step(dice)
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

    wandb.finish()