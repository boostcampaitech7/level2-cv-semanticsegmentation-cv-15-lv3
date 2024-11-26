import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import argparse
from datetime import timedelta
import time
import cv2
import random
from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from config.config import Config
from dataset.transforms import Transforms
from utils.metrics import dice_coef
from utils.rle import encode_mask_to_rle

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)  

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root=None, is_train=True, transforms=None):
        self.is_train = is_train
        self.transforms = transforms
        self.CLASS2IND = Config.CLASS2IND
        self.image_root = image_root
        self.label_root = label_root
        
        # Get PNG and JSON files
        self.pngs = self._get_pngs()
        self.jsons = self._get_jsons() if label_root else None
        
        if label_root:
            # Verify matching between pngs and jsons
            jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in self.jsons}
            pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in self.pngs}
            assert len(jsons_fn_prefix - pngs_fn_prefix) == 0, "Some JSON files don't have matching PNGs"
            assert len(pngs_fn_prefix - jsons_fn_prefix) == 0, "Some PNG files don't have matching JSONs"
        
        # Split dataset
        _filenames = np.array(self.pngs)
        _labelnames = np.array(self.jsons) if self.jsons else None

        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for _ in _filenames]
        
        # 전체 데이터의 20%를 validation data로 사용
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y]) if _labelnames is not None else []
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y]) if _labelnames is not None else []
                break
        
        self.filenames = filenames
        self.labelnames = labelnames

    def _get_pngs(self):
        return sorted([
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        ])
        
    def _get_jsons(self):
        return sorted([
            os.path.relpath(os.path.join(root, fname), start=self.label_root)
            for root, _dirs, files in os.walk(self.label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # (H, W, NC) 모양의 label 생성
        label_shape = tuple(image.shape[:2]) + (len(Config.CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일 읽기
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리
        for ann in annotations:
            c = ann["label"]
            class_ind = self.CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 변환
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.is_train else label
        
        # channel first 포맷으로 변경
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float(), torch.from_numpy(label).float(), os.path.basename(image_path)
    
class StratifiedXRayDataset(XRayDataset):
    def __init__(self, image_root, label_root=None, is_train=True, transforms=None, meta_path=None):
        self.is_train = is_train
        self.transforms = transforms
        self.CLASS2IND = Config.CLASS2IND
        self.image_root = image_root
        self.label_root = label_root
        
        # Get PNG and JSON files
        self.pngs = self._get_pngs()
        self.jsons = self._get_jsons() if label_root else None
        
        if label_root:
            # Verify matching between pngs and jsons
            jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in self.jsons}
            pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in self.pngs}
            assert len(jsons_fn_prefix - pngs_fn_prefix) == 0, "Some JSON files don't have matching PNGs"
            assert len(pngs_fn_prefix - jsons_fn_prefix) == 0, "Some PNG files don't have matching JSONs"
        
        # Load meta data
        self.meta_df = pd.read_excel(meta_path)
        self.meta_df = self.meta_df.drop('Unnamed: 5', axis=1)
        self.meta_df['ID'] = self.meta_df.index.map(lambda x: f"ID{str(x+1).zfill(3)}")
        self.meta_df['Gender'] = self.meta_df['성별'].apply(lambda x: 'Female' if '여' in str(x) else 'Male')
        self.meta_df = self.meta_df.rename(columns={'키(신장)': 'Height'})
        
        # Create height quartiles and strata
        self.meta_df['Height_Quartile'] = pd.qcut(self.meta_df['Height'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        self.meta_df['Strata'] = self.meta_df['Gender'] + '_' + self.meta_df['Height_Quartile'].astype(str)
        
        # Split dataset
        _filenames = np.array(self.pngs)
        _labelnames = np.array(self.jsons) if self.jsons else None
        
        # Get groups and strata
        groups = [os.path.dirname(fname) for fname in _filenames]
        image_strata = []
        for fname in _filenames:
            id_folder = os.path.dirname(fname).split('/')[-1]
            strata = self.meta_df[self.meta_df['ID'] == id_folder]['Strata'].iloc[0]
            image_strata.append(strata)
        
        # StratifiedGroupKFold 사용
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 첫 번째 fold를 사용
        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(_filenames, y=image_strata, groups=groups)):
            if fold_idx == 0:  # 첫 번째 fold만 사용
                if is_train:
                    filenames = list(_filenames[train_idx])
                    labelnames = list(_labelnames[train_idx]) if _labelnames is not None else []
                else:
                    filenames = list(_filenames[val_idx])
                    labelnames = list(_labelnames[val_idx]) if _labelnames is not None else []
                break
        
        self.filenames = filenames
        self.labelnames = labelnames

    def _get_pngs(self):
        # XRayDataset의 메서드 재사용
        return super()._get_pngs()
    
    def _get_jsons(self):
        # XRayDataset의 메서드 재사용
        return super()._get_jsons()
    
    def __len__(self):
        # XRayDataset의 메서드 재사용
        return super().__len__()
    
    def __getitem__(self, item):
        # XRayDataset의 메서드 재사용
        return super().__getitem__(item)

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
            for idx, (images, masks, image_names) in enumerate(data_loader):
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
                    image_name = image_names[b_idx]
                    
                    for c in range(masks_for_rle.shape[1]):
                        if model is not None:
                            pred_rle = encode_mask_to_rle(outputs_for_rle[b_idx, c])
                            pred_rles.append(pred_rle)
                        
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
        return pred_df  # gt_df 제거
    
    gt_df = pd.DataFrame({
        "image_name": filenames,
        "class": classes,
        "rle": gt_rles,
    })
    return gt_df  # pred_df 제거

def main(args):
    set_seed(Config.RANDOM_SEED)
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터셋 준비
    valid_dataset = StratifiedXRayDataset(
        image_root=Config.TRAIN_IMAGE_ROOT,
        label_root=Config.TRAIN_LABEL_ROOT,
        is_train=False,
        transforms=Transforms.get_valid_transform(),
        meta_path=Config.META_PATH
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
            pred_df = validation(model, valid_loader, device, args.threshold, save_gt=True)
            gt_df = validation(None, valid_loader, device, args.threshold, save_gt=True)
            model_name = args.model_path.split('/')[-1]
            pred_df.to_csv(f"{model_name.split('.')[0]}_val.csv", index=False)
            gt_df.to_csv("val_gt.csv", index=False)
            print(f"\nPrediction results saved to {model_name.split('.')[0]}_val.csv")
            print(f"Ground truth results saved to val_gt.csv")
        else:
            pred_df = validation(model, valid_loader, device, args.threshold)
            model_name = args.model_path.split('/')[-1]
            pred_df.to_csv(f"{model_name.split('.')[0]}_val.csv", index=False)
            print(f"\nResults saved to {model_name.split('.')[0]}_val.csv")
    else:
        # Ground truth만 생성
        print("\nGenerating ground truth only...")
        gt_df = validation(None, valid_loader, device, args.threshold, save_gt=True)
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