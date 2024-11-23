from Util.DiceCoef import dice_coef
import torch
from config import CLASSES
import torch.nn.functional as F
def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.cuda()
    model.eval()

    total_loss = 0
    total_focal = 0
    total_iou = 0
    total_dice = 0
    total_msssim = 0
    num_samples = 0  # 총 샘플 수 계산

    dices = []

    with torch.no_grad():
        total_steps = len(data_loader)

        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()

            # 모델 예측
            outputs = model(images)

            # 출력 크기 보정 (필요한 경우만)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            
            # 손실 계산
            batch_loss, batch_focal, batch_iou, batch_dice, batch_msssim = criterion(outputs, masks)
            total_loss += batch_loss.item()
            total_focal += batch_focal.item()
            total_iou += batch_iou.item()
            total_dice += batch_dice.item()
            total_msssim += batch_msssim.item()

            num_samples += 1

            # Dice 계산
            outputs = (outputs > thr).float()
            dice = dice_coef(outputs, masks)
            dices.append(dice.detach())

            # 진행 상황 출력
            if (step + 1) % 80 == 0 or (step + 1) == total_steps:
                print(f"Validation Progress: Step {step + 1}/{total_steps}")

    # Loss 평균 계산
    avg_loss = total_loss / num_samples
    avg_focal = total_focal / num_samples
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_msssim = total_msssim / num_samples

    # Dice 평균 계산
    dices = torch.cat(dices, 0)
    dices_per_class = dices.mean(dim=0)
    avg_dice_score = dices_per_class.mean().item()

    # 클래스별 Dice 점수 출력
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    # 최종 결과 출력
    print(f"Validation Completed: Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice_score:.4f}")
    print(f"Focal: {avg_focal:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}, MS-SSIM: {avg_msssim:.4f}")

    return avg_dice_score
