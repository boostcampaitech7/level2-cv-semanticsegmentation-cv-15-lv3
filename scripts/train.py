import datetime
import torch
from tqdm import tqdm
from config import MODEL_NAME, NUM_EPOCHS, CLASSES, SAVED_DIR, SERVER_ID, VAL_EVERY  # Add CLASSES to the import
from utils import save_model, dice_coef
import torch.nn.functional as F 
from discord_notifications import send_discord_message  # 추가

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

def train(model, data_loader, val_loader, criterion, optimizer):
    # 학습 시작 알림
    server_id = SERVER_ID
    
    send_discord_message(f"🎬 [서버 {server_id}] {MODEL_NAME} 학습이 시작되었습니다!")
    
    print(f'Start training..')
    model.cuda()

    n_class = len(CLASSES)
    best_dice = 0.

    for epoch in range(NUM_EPOCHS):
        model.train()
        
        # 10 에폭 시작시 현재 진행상황 알림
        if (epoch) % 10 == 0:
            send_discord_message(f"📊 [서버 {server_id}] 현재 진행상황: Epoch [{epoch+1}/{NUM_EPOCHS}] 진행 중")

        for step, (images, masks) in enumerate(data_loader):
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
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)

            if best_dice < dice:
                # send_discord_message(f"🎯 [서버 {server_id}] 새로운 최고 성능 달성!\n"
                #                    f"Epoch: {epoch + 1}/{NUM_EPOCHS} : "
                #                    f"이전 성능: {best_dice:.4f} -> 새로운 성능: {dice:.4f}\n"
                #                    )
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)

    # 학습 종료 알림
    send_discord_message(f"✨ [서버 {server_id}] 학습이 완료되었습니다!\n"
                        f"최종 최고 성능: {best_dice:.4f}")