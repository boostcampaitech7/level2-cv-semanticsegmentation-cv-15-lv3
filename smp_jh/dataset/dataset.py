import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold 
from config.config import Config

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
        
        return torch.from_numpy(image).float(), torch.from_numpy(label).float()
        

class XRayInferenceDataset(Dataset):
    def __init__(self, image_root, transforms=None):
        self.image_root = image_root
        self.transforms = transforms
        self.filenames = self._get_pngs()

    def _get_pngs(self):
        return sorted([
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).float(), image_name
    

# class StratifiedXRayDataset(XRayDataset):
#     def __init__(self, image_root, label_root=None, is_train=True, transforms=None, meta_path=None):
#         self.is_train = is_train
#         self.transforms = transforms
#         self.CLASS2IND = Config.CLASS2IND
#         self.image_root = image_root
#         self.label_root = label_root
        
#         # Get PNG and JSON files
#         self.pngs = self._get_pngs()
#         self.jsons = self._get_jsons() if label_root else None
        
#         if label_root:
#             # Verify matching between pngs and jsons
#             jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in self.jsons}
#             pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in self.pngs}
#             assert len(jsons_fn_prefix - pngs_fn_prefix) == 0, "Some JSON files don't have matching PNGs"
#             assert len(pngs_fn_prefix - jsons_fn_prefix) == 0, "Some PNG files don't have matching JSONs"
        
#         # Load meta data
#         self.meta_df = pd.read_excel(meta_path)
#         self.meta_df = self.meta_df.drop('Unnamed: 5', axis=1)
#         self.meta_df['ID'] = self.meta_df.index.map(lambda x: f"ID{str(x+1).zfill(3)}")
#         self.meta_df['Gender'] = self.meta_df['성별'].apply(lambda x: 'Female' if '여' in str(x) else 'Male')
#         self.meta_df = self.meta_df.rename(columns={'키(신장)': 'Height'})
        
#         # Create height quartiles and strata
#         self.meta_df['Height_Quartile'] = pd.qcut(self.meta_df['Height'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
#         self.meta_df['Strata'] = self.meta_df['Gender'] + '_' + self.meta_df['Height_Quartile'].astype(str)
        
#         # Split dataset
#         _filenames = np.array(self.pngs)
#         _labelnames = np.array(self.jsons) if self.jsons else None
        
#         # Get groups and strata
#         groups = [os.path.dirname(fname) for fname in _filenames]
#         image_strata = []
#         for fname in _filenames:
#             id_folder = os.path.dirname(fname).split('/')[-1]
#             strata = self.meta_df[self.meta_df['ID'] == id_folder]['Strata'].iloc[0]
#             image_strata.append(strata)
        
#         # Find best split
#         gkf = GroupKFold(n_splits=5)
#         best_split = None
#         best_score = float('inf')
        
#         # 각 fold의 점수를 저장할 리스트
#         fold_scores = []
        
#         for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(_filenames, groups=groups)):
#             # Calculate strata distribution
#             train_strata = [image_strata[i] for i in train_idx]
#             val_strata = [image_strata[i] for i in val_idx]
            
#             train_dist = pd.Series(train_strata).value_counts(normalize=True)
#             val_dist = pd.Series(val_strata).value_counts(normalize=True)
            
#             # Calculate distribution difference
#             score = 0
#             for strata in set(image_strata):
#                 train_prop = train_dist.get(strata, 0)
#                 val_prop = val_dist.get(strata, 0)
#                 score += (train_prop - val_prop) ** 2
            
#             fold_scores.append((fold_idx, score))
            
#             if score < best_score:
#                 best_score = score
#                 best_split = (train_idx, val_idx)
        
#         # 각 fold의 점수 출력
#         print("\nFold Scores:")
#         for fold_idx, score in sorted(fold_scores, key=lambda x: x[1]):
#             print(f"Fold {fold_idx}: {score:.6f}")
#         print(f"Selected best fold with score: {best_score:.6f}")
        
#         # Use best split
#         train_idx, val_idx = best_split
#         if is_train:
#             filenames = list(_filenames[train_idx])
#             labelnames = list(_labelnames[train_idx]) if _labelnames is not None else []
#         else:
#             filenames = list(_filenames[val_idx])
#             labelnames = list(_labelnames[val_idx]) if _labelnames is not None else []
        
#         self.filenames = filenames
#         self.labelnames = labelnames

#     def _get_pngs(self):
#         # XRayDataset의 메서드 재사용
#         return super()._get_pngs()
    
#     def _get_jsons(self):
#         # XRayDataset의 메서드 재사용
#         return super()._get_jsons()
    
#     def __len__(self):
#         # XRayDataset의 메서드 재사용
#         return super().__len__()
    
#     def __getitem__(self, item):
#         # XRayDataset의 메서드 재사용
#         return super().__getitem__(item)
    
#     # 추가 메서드들 (통계 관련)
#     def get_ids(self):
#         """현재 데이터셋의 모든 고유 ID 반환"""
#         return set([os.path.dirname(fname).split('/')[-1] for fname in self.filenames])
    
#     def get_gender_distribution(self):
#         """성별 분포 반환"""
#         ids = self.get_ids()
#         gender_dist = self.meta_df[self.meta_df['ID'].isin(ids)]['Gender'].value_counts(normalize=True)
#         return {
#             'Male': gender_dist.get('Male', 0),
#             'Female': gender_dist.get('Female', 0)
#         }
    
#     def get_height_distribution(self):
#         """키 분포 통계 반환"""
#         ids = self.get_ids()
#         heights = self.meta_df[self.meta_df['ID'].isin(ids)]['Height']
#         return {
#             'mean': heights.mean(),
#             'std': heights.std(),
#             'min': heights.min(),
#             'max': heights.max(),
#             'quartiles': {
#                 'Q1': heights.quantile(0.25),
#                 'Q2': heights.quantile(0.50),
#                 'Q3': heights.quantile(0.75)
#             }
#         }

#     def get_strata_distribution(self):
#         """층화 그룹 분포 반환"""
#         ids = self.get_ids()
#         return self.meta_df[self.meta_df['ID'].isin(ids)]['Strata'].value_counts(normalize=True).to_dict()

#     def print_dataset_stats(self):
#         """데이터셋 통계 출력"""
#         print(f"\n{'='*50}")
#         print(f"Dataset Statistics ({'Train' if self.is_train else 'Validation'}):")
#         print(f"{'='*50}")
#         print(f"Total samples: {len(self)}")
#         print(f"Unique patients: {len(self.get_ids())}")
        
#         print("\nGender Distribution:")
#         for gender, prop in self.get_gender_distribution().items():
#             print(f"{gender}: {prop:.1%}")
        
#         height_dist = self.get_height_distribution()
#         print(f"\nHeight Distribution:")
#         print(f"Mean ± Std: {height_dist['mean']:.1f} ± {height_dist['std']:.1f} cm")
#         print(f"Range: {height_dist['min']:.1f} - {height_dist['max']:.1f} cm")
#         print("\nHeight Quartiles:")
#         for q, val in height_dist['quartiles'].items():
#             print(f"{q}: {val:.1f} cm")
        
#         print("\nStrata Distribution:")
#         for strata, prop in self.get_strata_distribution().items():
#             print(f"{strata}: {prop:.1%}")
#         print(f"{'='*50}\n")


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
    
    # 추가 메서드들 (통계 관련)
    def get_ids(self):
        """현재 데이터셋의 모든 고유 ID 반환"""
        return set([os.path.dirname(fname).split('/')[-1] for fname in self.filenames])
    
    def get_gender_distribution(self):
        """성별 분포 반환"""
        ids = self.get_ids()
        gender_dist = self.meta_df[self.meta_df['ID'].isin(ids)]['Gender'].value_counts(normalize=True)
        return {
            'Male': gender_dist.get('Male', 0),
            'Female': gender_dist.get('Female', 0)
        }
    
    def get_height_distribution(self):
        """키 분포 통계 반환"""
        ids = self.get_ids()
        heights = self.meta_df[self.meta_df['ID'].isin(ids)]['Height']
        return {
            'mean': heights.mean(),
            'std': heights.std(),
            'min': heights.min(),
            'max': heights.max(),
            'quartiles': {
                'Q1': heights.quantile(0.25),
                'Q2': heights.quantile(0.50),
                'Q3': heights.quantile(0.75)
            }
        }

    def get_strata_distribution(self):
        """층화 그룹 분포 반환"""
        ids = self.get_ids()
        return self.meta_df[self.meta_df['ID'].isin(ids)]['Strata'].value_counts(normalize=True).to_dict()

    def print_dataset_stats(self):
        """데이터셋 통계 출력"""
        print(f"\n{'='*50}")
        print(f"Dataset Statistics ({'Train' if self.is_train else 'Validation'}):")
        print(f"{'='*50}")
        print(f"Total samples: {len(self)}")
        print(f"Unique patients: {len(self.get_ids())}")
        
        print("\nGender Distribution:")
        for gender, prop in self.get_gender_distribution().items():
            print(f"{gender}: {prop:.1%}")
        
        height_dist = self.get_height_distribution()
        print(f"\nHeight Distribution:")
        print(f"Mean ± Std: {height_dist['mean']:.1f} ± {height_dist['std']:.1f} cm")
        print(f"Range: {height_dist['min']:.1f} - {height_dist['max']:.1f} cm")
        print("\nHeight Quartiles:")
        for q, val in height_dist['quartiles'].items():
            print(f"{q}: {val:.1f} cm")
        
        print("\nStrata Distribution:")
        for strata, prop in self.get_strata_distribution().items():
            print(f"{strata}: {prop:.1%}")
        print(f"{'='*50}\n")