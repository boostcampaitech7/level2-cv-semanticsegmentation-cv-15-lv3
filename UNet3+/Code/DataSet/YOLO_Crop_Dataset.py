import os
import cv2
import numpy as np
import json
import torch
from config import CLASS2IND, CLASSES, IMAGE_ROOT, LABEL_ROOT,YOLO_NAMES,YOLO_SELECT_CLASS,IMSIZE
from Util.SetSeed import set_seed

set_seed()

from torch.utils.data import Dataset

class XRayDataset(Dataset):
    def __init__(self, filenames, labelnames, transforms=None, is_train=False, yolo_model=None):
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        self.yolo_model = yolo_model  # YOLO 모델 추가

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
        for ann in annotations:
            c = ann["label"]
            if c not in CLASSES:
                continue

            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # Generate masks
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        # YOLO 예측 결과에서 others 클래스 박스 가져오기
        if self.yolo_model:
            results = self.yolo_model.predict(image_path, imgsz=2048, iou=0.3, conf=0.1, max_det=3)
            result=results[0].boxes
            yolo_boxes = result.xyxy.cpu().numpy()  # (N, 4) 형식의 박스 좌표
            yolo_classes = result.cls.cpu().numpy()  # (N,) 형식의 클래스
            yolo_confidences = result.conf.cpu().numpy()  # (N,) 형식의 신뢰도

            # others 클래스 필터링
            others_boxes = [
                (box, conf) for box, cls, conf in zip(yolo_boxes, yolo_classes, yolo_confidences)
                if YOLO_NAMES[int(cls)] == YOLO_SELECT_CLASS
            ]

            # 신뢰도가 가장 높은 박스 선택
            if others_boxes:
                best_box, _ = max(others_boxes, key=lambda x: x[1])  # (x1, y1, x2, y2) 좌표
                crop_box = self.calculate_crop_box_from_yolo(best_box, image.shape[:2])
                image = self.crop_image(image, crop_box)
                label = self.crop_label(label, crop_box)
            
        # Apply augmentations
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # Convert to tensor
        image = image.transpose(2, 0, 1)  # Channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

    def calculate_crop_box_from_yolo(self, yolo_box, image_size, crop_size=IMSIZE):
        """Calculate the crop box based on YOLO prediction."""
        x1, y1, x2, y2 = yolo_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        half_size = crop_size / 2
        start_x = max(int(center_x - half_size), 0)
        start_y = max(int(center_y - half_size), 0)
        end_x = min(int(start_x + crop_size), image_size[1])
        end_y = min(int(start_y + crop_size), image_size[0])

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
