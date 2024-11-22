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
    model.eval()
    
    dices = []
    pred_rles = []
    gt_rles = []  # ground truth RLEs
    filename_and_class = []
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc='[Validation]') as pbar:
            for idx, (images, masks) in enumerate(data_loader):
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
                
                # Ground truth도 2048x2048로 리사이즈
                if save_gt:
                    masks_for_rle = F.interpolate(masks.float(), size=(2048, 2048), mode="nearest")
                    masks_for_rle = masks_for_rle.bool().cpu().numpy()
                
                # RLE 인코딩
                for b_idx, (output, mask) in enumerate(zip(outputs_for_rle, masks_for_rle if save_gt else outputs_for_rle)):
                    image_name = f"image_{idx * data_loader.batch_size + b_idx}"
                    for c, pred in enumerate(output):
                        # 예측값 RLE
                        pred_rle = encode_mask_to_rle(pred)
                        pred_rles.append(pred_rle)
                        filename_and_class.append(f"{Config.IND2CLASS[c]}_{image_name}")
                        
                        # Ground truth RLE
                        if save_gt:
                            gt_rle = encode_mask_to_rle(mask[c])
                            gt_rles.append(gt_rle)
                
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
    
    # Create DataFrame for prediction results
    classes, filenames = zip(*[x.split("_", 1) for x in filename_and_class])
    pred_df = pd.DataFrame({
        "image_name": filenames,
        "class": classes,
        "rle": pred_rles,
    })
    
    # Create DataFrame for ground truth if requested
    if save_gt:
        gt_df = pd.DataFrame({
            "image_name": filenames,
            "class": classes,
            "rle": gt_rles,
        })
        return pred_df, gt_df
    
    return pred_df

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
    
    # Validation 실행 및 결과 저장
    if args.save_gt:
        pred_df, gt_df = validation(model, valid_loader, device, args.threshold, save_gt=True)
        model_name = args.model_path.split('/')[-1]
        pred_df.to_csv(f"{model_name.split('.')[0]}_pred.csv", index=False)
        gt_df.to_csv("val_gt.csv", index=False)
        print(f"\nPrediction results saved to {model_name}_pred.csv")
        print(f"Ground truth results saved to {model_name}_gt.csv")
    else:
        pred_df = validation(model, valid_loader, device, args.threshold)
        pred_df.to_csv(f"{args.model_path.split('/')[-1]}_val.csv", index=False)
        print(f"\nResults saved to {args.model_path.split('/')[-1]}_val.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Validation batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    parser.add_argument('--save_gt', action='store_true',
                        help='Save ground truth masks as separate CSV')
    
    args = parser.parse_args()
    main(args)