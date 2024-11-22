from Util.DiceCoef import dice_coef
import torch
from config import CLASSES
import torch.nn.functional as F

def validation(epoch, model, data_loader, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.cuda()
    model.eval()

    dices = []

    with torch.no_grad():
        total_steps = len(data_loader)  # 데이터 로더 총 스텝 수

        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()

            # 모델 예측
            outputs = model(images)

            # 출력 크기 보정 (필요한 경우만)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            
            # 출력 이진화 및 Dice 계산 (GPU 상에서 처리)
            outputs = (outputs > thr).float()
            dice = dice_coef(outputs, masks)
            dices.append(dice.detach())  # GPU에서 유지

            # 진행 상황 출력
            if (step + 1) % 80 == 0 or (step + 1) == total_steps:  # 매 80 스텝마다 또는 마지막 스텝에서 출력
                print(f"Validation Progress: Step {step + 1}/{total_steps}")

    # GPU 상에서 Dice 평균 계산
    dices = torch.cat(dices, 0)
    dices_per_class = dices.mean(dim=0)
    
    # 클래스별 Dice 점수 출력
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    avg_dice = dices_per_class.mean().item()

    # 최종 평균 Dice 출력
    print(f"Validation Completed: Avg Dice: {avg_dice:.4f}")

    return avg_dice
