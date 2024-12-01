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
    
    # train_set & validation_set list를 txt 파일로 저장하기 위한 코드
    def save_file_lists(self, output_dir="output_files"):
        """
        Train과 Validation 파일 리스트를 텍스트 파일로 저장
        Args:
            output_dir (str): 텍스트 파일을 저장할 디렉토리 경로
        """
        os.makedirs(output_dir, exist_ok=True)  # 저장할 폴더 생성

        # Train 파일 저장
        if self.is_train:
            train_file_path = os.path.join(output_dir, "train_files.txt")
            with open(train_file_path, "w") as f:
                for file_name in self.filenames:
                    f.write(f"{file_name}\n")
            print(f"Train file list saved to {train_file_path}")

        # Validation 파일 저장
        else:
            val_file_path = os.path.join(output_dir, "val_files.txt")
            with open(val_file_path, "w") as f:
                for file_name in self.filenames:
                    f.write(f"{file_name}\n")
            print(f"Validation file list saved to {val_file_path}")

    def get_image_info(self, id_folder, image_name=None):
        """
        특정 ID 폴더의 이미지 정보를 반환
        Args:
            id_folder (str): ID 폴더명 (예: 'ID001')
            image_name (str, optional): 특정 이미지 파일명. None이면 해당 ID의 모든 이미지 반환
        Returns:
            dict: 이미지 정보를 담은 딕셔너리
        """
        matching_files = []
        for fname in self.filenames:
            if id_folder in fname:
                if image_name is None or image_name in fname:
                    matching_files.append({
                        'image_path': os.path.join(self.image_root, fname),
                        'label_path': os.path.join(self.label_root, self.labelnames[self.filenames.index(fname)])
                        if self.label_root else None
                    })
        
        return matching_files

    def clean_image_annotations(self, id_folder, image_name=None):
        """
        특정 ID 폴더의 이미지 어노테이션을 검사하고 문제점을 보고
        Args:
            id_folder (str): ID 폴더명 (예: 'ID001')
            image_name (str, optional): 특정 이미지 파일명
        """
        images = self.get_image_info(id_folder, image_name)
        
        for img_info in images:
            print(f"\nChecking: {img_info['image_path']}")
            
            # 이미지 체크
            if not os.path.exists(img_info['image_path']):
                print(f"Warning: Image file does not exist")
                continue
                
            image = cv2.imread(img_info['image_path'])
            if image is None:
                print(f"Error: Cannot read image file")
                continue
                
            # 라벨 체크
            if img_info['label_path'] and os.path.exists(img_info['label_path']):
                with open(img_info['label_path'], 'r') as f:
                    try:
                        annotations = json.load(f)['annotations']
                        
                        # 어노테이션 유효성 검사
                        for ann in annotations:
                            if 'label' not in ann:
                                print(f"Error: Missing label in annotation")
                            if 'points' not in ann:
                                print(f"Error: Missing points in annotation")
                            else:
                                points = np.array(ann['points'])
                                if points.size == 0:
                                    print(f"Error: Empty points array for label {ann.get('label', 'unknown')}")
                                if points.shape[1] != 2:
                                    print(f"Error: Invalid points format for label {ann.get('label', 'unknown')}")
                            
                            # 클래스 확인
                            if ann.get('label') not in self.CLASS2IND:
                                print(f"Error: Invalid class label: {ann.get('label')}")
                        
                        print(f"Found {len(annotations)} valid annotations")
                        
                    except json.JSONDecodeError:
                        print(f"Error: Invalid JSON format in label file")
            else:
                print(f"Warning: Label file not found")

    def fix_specific_annotation(self, id_folder="ID058", image_name="image1661392103627.png"):
        """
        특정 이미지의 어노테이션 수정
        """
        # 라벨 수정 매핑
        label_mapping = {
            'finger-1': 'finger-3',
            'finger-3': 'finger-1',
            'Hamate': 'Trapezium',
            'Capitate': 'Trapezoid',
            'Trapezoid': 'Capitate',
            'Trapezium': 'Hamate',
            'Pisiform': 'Scaphoid',
            'Triquetrum': 'Lunate',
            'Scaphoid': 'Triquetrum',
            'Lunate': 'Pisiform',
            'Radius': 'Ulna',
            'Ulna': 'Radius'
        }

        # 이미지 정보 가져오기
        images = self.get_image_info(id_folder, image_name)
        
        if not images:
            print(f"Error: Image not found for {id_folder}/{image_name}")
            return
        
        label_path = images[0]['label_path']
        
        if not os.path.exists(label_path):
            print(f"Error: Label file not found at {label_path}")
            return
        
        # 백업 파일 생성
        backup_path = label_path + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(label_path, backup_path)
            print(f"Created backup at {backup_path}")
        
        # 현재 라벨 확인을 위한 디버깅 출력 추가
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
                print("\nCurrent labels in the file:")
                current_labels = set(ann['label'] for ann in data['annotations'])
                print(current_labels)
                
                # 어노테이션 수정
                modified = False
                for ann in data['annotations']:
                    old_label = ann['label']
                    if old_label in label_mapping:
                        print(f"Changing label: {old_label} -> {label_mapping[old_label]}")
                        ann['label'] = label_mapping[old_label]
                        modified = True
                
                if modified:
                    # 수정된 내용 저장
                    with open(label_path, 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"Successfully updated annotations in {label_path}")
                else:
                    print("No modifications were needed")
                    
        except Exception as e:
            print(f"Error occurred while updating annotations: {str(e)}")
            if os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, label_path)
                print("Restored from backup due to error")

    def remove_specific_mask(self, id_folder="ID363", image_name="image1664935962797.png", label_to_remove="finger-14"):
        """
        특정 이미지에서 특정 라벨의 마스크를 제거
        Args:
            id_folder (str): ID 폴더명
            image_name (str): 이미지 파일명
            label_to_remove (str): 제거할 라벨명
        """
        # 이미지 정보 가져오기
        images = self.get_image_info(id_folder, image_name)
        
        if not images:
            print(f"Error: Image not found for {id_folder}/{image_name}")
            return
        
        label_path = images[0]['label_path']
        
        if not os.path.exists(label_path):
            print(f"Error: Label file not found at {label_path}")
            return
        
        # 백업 파일 생성
        backup_path = label_path + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(label_path, backup_path)
            print(f"Created backup at {backup_path}")
        
        try:
            # JSON 파일 읽기
            with open(label_path, 'r') as f:
                data = json.load(f)
            
            # 기존 어노테이션 개수
            original_count = len(data['annotations'])
            
            # 특정 라벨 제거
            data['annotations'] = [
                ann for ann in data['annotations'] 
                if ann['label'] != label_to_remove
            ]
            
            # 제거된 어노테이션 개수
            removed_count = original_count - len(data['annotations'])
            
            if removed_count > 0:
                # 수정된 내용 저장
                with open(label_path, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"Successfully removed {removed_count} annotations with label '{label_to_remove}'")
            else:
                print(f"No annotations found with label '{label_to_remove}'")
                
        except Exception as e:
            print(f"Error occurred while updating annotations: {str(e)}")
            # 에러 발생시 백업에서 복구
            if os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, label_path)
                print("Restored from backup due to error")
