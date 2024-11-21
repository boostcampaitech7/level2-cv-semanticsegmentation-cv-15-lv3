import os
import cv2
import numpy as np
import json
import torch
from config import CLASS2IND, CLASSES, IMAGE_ROOT, LABEL_ROOT,IMSIZE
from Util.SetSeed import set_seed

set_seed()

from torch.utils.data import Dataset
class XRayDataset(Dataset):
    def __init__(self, filenames, labelnames, transforms=None,
                 is_train=False, save_dir=None, draw_enabled=False):
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        self.save_dir = save_dir  # Crop된 이미지 저장 디렉토리
        self.draw_enabled = draw_enabled  # 라벨 그리기 기능 활성화 여부

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        # Read and normalize image
        image = cv2.imread(image_path)
        image = image / 255.0

        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # Initialize label tensor
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # Read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # Generate masks for all annotations
        points = []
        for ann in annotations:
            c = ann["label"]
            if c not in CLASSES:
                continue

            class_ind = CLASS2IND[c]
            points.extend(ann["points"])  # 라벨 데이터에서 모든 Points를 수집

            # Generate masks
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [np.array(ann["points"])], 1)
            label[..., class_ind] = class_label
        
        # Points 기반 크롭 영역 계산
        if points:
            crop_box = self.calculate_crop_box_from_points(points, image.shape[:2])
            image = self.crop_image(image, crop_box)
            label = self.crop_label(label, crop_box)
            
        # Apply augmentations
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.is_train else label

        if self.draw_enabled and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"cropped_{os.path.basename(self.filenames[item])}")
            draw_and_save_crop(image, label, save_path)
        
        # Convert to tensor
        image = image.transpose(2, 0, 1)  # Channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        
        
        return image, label

    def calculate_crop_box_from_points(self, points, image_size, crop_size=IMSIZE):
        """Points에서 크롭 박스를 계산합니다."""
        points = np.array(points)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)

        # 중심점 계산
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # 크롭 영역 계산
        half_size = crop_size / 2
        start_x = max(int(center_x - half_size), 0)
        start_y = max(int(center_y - half_size), 0)
        end_x = min(start_x + crop_size, image_size[1])
        end_y = min(start_y + crop_size, image_size[0])

        return start_x, start_y, end_x, end_y

    def crop_image(self, image, crop_box):
        """Crop the image to the specified box."""
        start_x, start_y, end_x, end_y = crop_box
        cropped_image = image[start_y:end_y, start_x:end_x]
        return cropped_image
    
    def crop_label(self, label, crop_box):
        """Crop the label tensor to match the cropped image."""
        start_x, start_y, end_x, end_y = crop_box
        return label[start_y:end_y, start_x:end_x, :]


import cv2
import numpy as np

def draw_and_save_crop(image, label, save_path):
    """
    Crop된 이미지 위에 라벨 정보를 그려 저장합니다.

    Args:
        image (np.ndarray): Crop된 이미지 (H, W, C).
        label (np.ndarray): Crop된 라벨 (H, W, num_classes).
        save_path (str): 저장할 파일 경로.
    """
    # 이미지 복사
    image_to_draw = (image * 255).astype(np.uint8).copy()  # 이미지 복원 (0~255)

    # 클래스별 색상 설정
    num_classes = label.shape[-1]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # 클래스별 색상

    # 클래스별로 라벨을 이미지에 그리기
    for class_idx in range(num_classes):
        mask = label[..., class_idx].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 컨투어 그리기
        for contour in contours:
            cv2.drawContours(image_to_draw, [contour], -1, colors[class_idx % len(colors)], 2)

    # 저장
    cv2.imwrite(save_path, image_to_draw)
