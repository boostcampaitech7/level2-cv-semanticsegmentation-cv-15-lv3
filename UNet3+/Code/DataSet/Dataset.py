import os
import cv2
import numpy as np
import json
import torch
from config import CLASS2IND, CLASSES, IMAGE_ROOT, LABEL_ROOT
from Util.SetSeed import set_seed

set_seed()

from torch.utils.data import Dataset
class XRayDataset(Dataset):
    def __init__(self, filenames, labelnames, transforms=None, is_train=False):
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)

        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # iterate each class
        for ann in annotations:
            c = ann["label"]
            
            if c not in CLASSES:
                continue
            
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label
    
class XRayDatasetCrop(Dataset):
    def __init__(self, filenames, labelnames, transforms=None, is_train=False):
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        # 이미지 로드 및 정규화
        image = cv2.imread(image_path)
        image = image / 255.

        # 작은 뼈 영역 크롭 (여유 마진 20픽셀 추가)
        margin = 20
        x_min, y_min = max(540 - margin, 0), max(811 - margin, 0)
        x_max, y_max = min(1505 + margin, image.shape[1]), min(1882 + margin, image.shape[0])
        
        # 이미지 크롭
        image = image[y_min:y_max, x_min:x_max]

        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)

        # (H, W, NC) 모양의 label 생성
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # label 파일 읽기
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # 클래스별 처리
        for ann in annotations:
            c = ann["label"]
            
            if c not in CLASSES:
                continue    
            
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # 포인트 좌표 조정 (크롭에 맞춰서)
            points[:, 0] = points[:, 0] - x_min
            points[:, 1] = points[:, 1] - y_min

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

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

class XRayDatasetCropAug(Dataset):
    def __init__(self, filenames, labelnames, transforms=None, is_train=False):
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        
        # CLAHE와 감마 보정 설정
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.gamma_value = 1.8  # 감마 값을 1.8로 설정

    def apply_image_enhancement(self, image):
        """CLAHE와 감마 보정을 적용하는 함수"""
        # BGR to LAB 변환 (CLAHE는 밝기 채널에만 적용)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE 적용
        l = self.clahe.apply(l)
        
        # LAB 채널 합치기
        lab = cv2.merge((l, a, b))
        
        # LAB to BGR 변환
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 감마 보정 적용
        enhanced = np.array(enhanced / 255.0, dtype=np.float32)
        enhanced = np.power(enhanced, self.gamma_value)
        enhanced = np.clip(enhanced * 255.0, 0, 255)
        enhanced = enhanced.astype(np.uint8)
        
        return enhanced

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        # 이미지 로드
        image = cv2.imread(image_path)
        
        # 이미지 개선 적용 (CLAHE + 감마 보정)
        image = self.apply_image_enhancement(image)
        
        # 정규화
        image = image / 255.

        # 작은 뼈 영역 크롭
        margin = 20
        x_min, y_min = max(540 - margin, 0), max(811 - margin, 0)
        x_max, y_max = min(1505 + margin, image.shape[1]), min(1882 + margin, image.shape[0])
        
        # 이미지 크롭
        image = image[y_min:y_max, x_min:x_max]

        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)

        # (H, W, NC) 모양의 label 생성
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # label 파일 읽기
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # 클래스별 처리
        for ann in annotations:
            c = ann["label"]
            
            if c not in CLASSES:
                continue    
            
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # 포인트 좌표 조정
            points[:, 0] = points[:, 0] - x_min
            points[:, 1] = points[:, 1] - y_min

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

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label