import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from config import TRAIN_IMAGE_ROOT, TRAIN_LABEL_ROOT, CLASSES, CLASS2IND, \
    train_jsons, train_pngs, TEST_IMAGE_ROOT, test_pngs

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, gamma_value=None):
        _filenames = np.array(train_pngs)
        _labelnames = np.array(train_jsons)

        # Train-validation split
        groups = [os.path.dirname(fname) for fname in _filenames]
        ys = [0 for fname in _filenames]

        gkf = GroupKFold(n_splits=5)
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                if i == 0:
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                break

        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        self.gamma_value = gamma_value  # 감마 값 설정

    def __len__(self):
        return len(self.filenames)
    
    def gamma_correction(self, image, gamma):
        """Apply gamma correction to the image."""
        image = np.array(image / 255.0, dtype=float)
        corrected_image = np.power(image, gamma)
        corrected_image = np.uint8(corrected_image * 255)
        return corrected_image
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(TRAIN_IMAGE_ROOT, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0  # 1채널 이미지로 읽기

        # 감마 보정 적용 (gamma_value가 지정된 경우)
        if self.gamma_value is not None:
            image = self.gamma_correction(image, self.gamma_value)

        # 이미지 차원 확장 (1채널 -> 3채널로 변환)
        image = np.expand_dims(image, axis=-1)  # (H, W, 1)
        image = np.repeat(image, 3, axis=-1)  # (H, W, 3)로 변환

        label_name = self.labelnames[item]
        label_path = os.path.join(TRAIN_LABEL_ROOT, label_name)

        label_shape = tuple(image.shape[:2]) + (len(CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image.transpose(2, 0, 1)  # channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = test_pngs
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(TEST_IMAGE_ROOT, image_name)

        image = cv2.imread(image_path) / 255.0

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        image = image.transpose(2, 0, 1)  # channel first
        image = torch.from_numpy(image).float()

        return image, image_name