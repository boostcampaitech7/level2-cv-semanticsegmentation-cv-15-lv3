# validate.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datetime import timedelta
import time

from config.config import Config
from dataset.dataset import XRayDataset
from dataset.transforms import Transforms
from utils.metrics import dice_coef

def validation(model, data_loader, device, threshold=0.5):
    """Validation 함수"""
    val_start = time.time()
    model.eval()
    
    dices = []
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc='[Validation]') as pbar:
            for images, masks in data_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Resize if needed
                output_h, output_w = outputs.size(-2), outputs.size(-1)
                mask_h, mask_w = masks.size(-2), masks.size(-1)
                if output_h != mask_h or output_w != mask_w:
                    outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                
                # Calculate dice score
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > threshold)
                dice = dice_coef(outputs, masks)
                dices.append(dice.detach().cpu())
                
                pbar.update(1)
                pbar.set_postfix(dice=torch.mean(dice).item())
    
    # Calculate metrics
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    
    # Print class-wise scores
    print("\nClass-wise Dice Scores:")
    print("-" * 30)
    for c, d in zip(Config.CLASSES, dices_per_class):
        print(f"{c:<12}: {d.item():.4f}")
    
    # Print summary
    avg_dice = torch.mean(dices_per_class).item()
    print("-" * 30)
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Validation Time: {timedelta(seconds=time.time()-val_start)}")
    
    return avg_dice, {c: d.item() for c, d in zip(Config.CLASSES, dices_per_class)}

def main(args):
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터셋 준비
    valid_dataset = XRayDataset(
        image_root=Config.TRAIN_IMAGE_ROOT,
        label_root=Config.TRAIN_LABEL_ROOT,
        is_train=False,
        transforms=Transforms.get_valid_transform()
    )
    
    # DataLoader
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 모델 로드
    model = torch.load(args.model_path)
    model = model.to(device)
    
    # Validation 실행
    avg_dice, class_dices = validation(model, valid_loader, device, args.threshold)
    
    # 결과 저장
    if args.save_results:
        import json
        results = {
            "average_dice": avg_dice,
            "class_dice_scores": class_dices
        }
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Validation batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    parser.add_argument('--save_results', type=str, default='',
                        help='Path to save validation results as JSON')
    
    args = parser.parse_args()
    main(args)