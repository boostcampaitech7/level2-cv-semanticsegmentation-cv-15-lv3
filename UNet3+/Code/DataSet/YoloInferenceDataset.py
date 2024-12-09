import os
import cv2
import numpy as np
import json
import torch
from config import CLASS2IND, CLASSES, IMAGE_ROOT, LABEL_ROOT,YOLO_NAMES,YOLO_SELECT_CLASS,IMSIZE,TEST_IMAGE_ROOT
from Util.SetSeed import set_seed

set_seed()

from torch.utils.data import Dataset

class XRayInferenceDataset(Dataset):
    def __init__(self, filenames,yolo_model, transforms=None, save_dir=None, draw_enabled=False):
        _filenames = filenames
        _filenames = np.array(sorted(_filenames))
        self.yolo_model=yolo_model
        self.filenames = _filenames
        self.transforms = transforms
        self.save_dir = save_dir  
        self.draw_enabled = draw_enabled  

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(TEST_IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        print(image.shape)
        
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
                print(crop_box,"@@@@")
                
        image = image / 255.
        
        '''if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]'''
        
        '''if self.draw_enabled and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"cropped_{os.path.basename(self.filenames[item])}")
            self.save_crop(image, save_path)'''

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first

        image = torch.from_numpy(image).float()

        return image, image_name, crop_box

    
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
        print(start_x, start_y, end_x, end_y)

        return start_x, start_y, end_x, end_y

    def crop_image(self, image, crop_box):
        """Crop the image to the specified box."""
        start_x, start_y, end_x, end_y = crop_box
        cropped_image = image[start_y:end_y, start_x:end_x]
        return cropped_image
    def save_crop(self,image,save_path):
        # 이미지 복사
        image_to_draw = (image * 255).astype(np.uint8).copy()  # 이미지 복원 (0~255)
        # 저장
        cv2.imwrite(save_path, image_to_draw)
