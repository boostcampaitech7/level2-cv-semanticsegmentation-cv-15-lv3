import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import argparse
from datetime import timedelta
import time

from config.config import Config
from dataset.dataset import XRayDataset
from dataset.transforms import Transforms
from utils.metrics import dice_coef
from utils.rle import encode_mask_to_rle

def validation(model, data_loader, device, threshold=0.5, save_gt=False):
    """Validation 함수"""
    val_start = time.time()
    if model is not None:
        model.eval()
    
    dices = []
    pred_rles = []
    gt_rles = []
    filename_and_class = []
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc='[Validation]') as pbar:
            for idx, (images, masks) in enumerate(data_loader):
                if model is not None:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    
                    # Resize for dice calculation
                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                    
                    # Calculate dice score
                    outputs_sigmoid = torch.sigmoid(outputs)
                    outputs_binary = (outputs_sigmoid > threshold)
                    dice = dice_coef(outputs_binary, masks)
                    dices.append(dice.detach().cpu())
                    
                    # Resize for RLE (2048x2048)
                    outputs_for_rle = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                    outputs_for_rle = torch.sigmoid(outputs_for_rle)
                    outputs_for_rle = (outputs_for_rle > threshold).detach().cpu().numpy()
                
                # Ground truth 처리
                masks = masks.to(device) if not masks.is_cuda else masks
                masks_for_rle = F.interpolate(masks.float(), size=(2048, 2048), mode="nearest")
                masks_for_rle = masks_for_rle.bool().cpu().numpy()
                
                # RLE 인코딩
                for b_idx in range(masks_for_rle.shape[0]):
                    image_name = f"image_{idx * data_loader.batch_size + b_idx}"
                    for c in range(masks_for_rle.shape[1]):
                        if model is not None:
                            # 예측값 RLE
                            pred_rle = encode_mask_to_rle(outputs_for_rle[b_idx, c])
                            pred_rles.append(pred_rle)
                        
                        # Ground truth RLE
                        gt_rle = encode_mask_to_rle(masks_for_rle[b_idx, c])
                        gt_rles.append(gt_rle)
                        filename_and_class.append(f"{Config.IND2CLASS[c]}_{image_name}")
                
                pbar.update(1)
                if model is not None:
                    pbar.set_postfix(dice=torch.mean(dice).item())
    
    if model is not None:
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
    
    # Create DataFrames
    classes, filenames = zip(*[x.split("_", 1) for x in filename_and_class])
    if model is not None:
        pred_df = pd.DataFrame({
            "image_name": filenames,
            "class": classes,
            "rle": pred_rles,
        })
    
    gt_df = pd.DataFrame({
        "image_name": filenames,
        "class": classes,
        "rle": gt_rles,
    })
    
    if model is not None:
        return pred_df, gt_df
    else:
        return None, gt_df

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
    
    if args.model_path:
        # 모델 로드
        model = torch.load(args.model_path)
        model = model.to(device)
        
        # Validation 실행 및 결과 저장
        if args.save_gt:
            pred_df, gt_df = validation(model, valid_loader, device, args.threshold, save_gt=True)
            model_name = args.model_path.split('/')[-1]
            pred_df.to_csv(f"{model_name.split('.')[0]}_val.csv", index=False)
            gt_df.to_csv("val_gt.csv", index=False)
            print(f"\nPrediction results saved to {model_name.split('.')[0]}_val.csv")
            print(f"Ground truth results saved to val_gt.csv")
        else:
            pred_df = validation(model, valid_loader, device, args.threshold)
            pred_df.to_csv(f"{args.model_path.split('/')[-1]}_val.csv", index=False)
            print(f"\nResults saved to {args.model_path.split('/')[-1]}_val.csv")
    else:
        # Ground truth만 생성
        print("\nGenerating ground truth only...")
        _, gt_df = validation(None, valid_loader, device, args.threshold, save_gt=True)
        gt_df.to_csv("val_gt.csv", index=False)
        print("Ground truth results saved to val_gt.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model checkpoint (optional)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Validation batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    parser.add_argument('--save_gt', action='store_true',
                        help='Save ground truth masks as separate CSV')
    
    args = parser.parse_args()
    main(args)