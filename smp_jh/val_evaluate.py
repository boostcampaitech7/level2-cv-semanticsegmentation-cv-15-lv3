import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from utils.rle import decode_rle_to_mask

def calculate_dice_score(pred_mask, gt_mask):
    """
    두 마스크 간의 dice score 계산
    """
    intersection = np.sum(pred_mask * gt_mask)
    if (np.sum(pred_mask) == 0) and (np.sum(gt_mask) == 0):
        return 1.0
    return (2 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)

def evaluate_predictions(pred_csv, gt_csv):
    """
    예측 결과와 ground truth CSV 파일을 비교하여 dice score 계산
    """
    # CSV 파일 로드
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)
    
    # 클래스별 dice scores 저장
    class_dices = {}
    
    # 각 클래스별로 처리
    for class_name in pred_df['class'].unique():
        class_dices[class_name] = []
        
        # 해당 클래스의 예측과 GT 추출
        pred_class = pred_df[pred_df['class'] == class_name]
        gt_class = gt_df[gt_df['class'] == class_name]
        
        # 각 이미지에 대해 처리
        for img_name in tqdm(pred_class['image_name'].unique(), 
                           desc=f'Processing {class_name}'):
            # 예측과 GT의 RLE 가져오기
            pred_rle = pred_class[pred_class['image_name'] == img_name]['rle'].values[0]
            gt_rle = gt_class[gt_class['image_name'] == img_name]['rle'].values[0]
            
            # RLE를 마스크로 디코딩
            pred_mask = decode_rle_to_mask(pred_rle, 2048, 2048)
            gt_mask = decode_rle_to_mask(gt_rle, 2048, 2048)
            
            # Dice score 계산
            dice = calculate_dice_score(pred_mask, gt_mask)
            class_dices[class_name].append(dice)
    
    # 결과 출력
    print("\nClass-wise Dice Scores:")
    print("-" * 30)
    
    total_dice = 0
    for class_name, dices in class_dices.items():
        avg_dice = np.mean(dices)
        total_dice += avg_dice
        print(f"{class_name:<12}: {avg_dice:.4f}")
    
    print("-" * 30)
    print(f"Average Dice: {total_dice/len(class_dices):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_csv', type=str, required=True,
                        help='Path to prediction CSV file')
    parser.add_argument('--gt_csv', type=str, required=True,
                        help='Path to ground truth CSV file')
    
    args = parser.parse_args()
    evaluate_predictions(args.pred_csv, args.gt_csv)

if __name__ == "__main__":
    main()