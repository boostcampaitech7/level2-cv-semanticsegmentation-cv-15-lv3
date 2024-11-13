import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# 데이터 경로를 입력하세요
IMAGE_ROOT = "../data/train/DCM"
LABEL_ROOT = "../data/train/outputs_json"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

class XRayDataset(Dataset):
    def __init__(self, filenames, labelnames, transforms=None, is_train=False):
        """
        Args:
            filenames (list): List of image file names.
            labelnames (list): List of corresponding label file names.
            transforms (callable, optional): Optional transformation to be applied on a sample.
            is_train (bool, optional): Flag indicating whether it's for training or validation.
        """
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, item):
        """
        Args:
            item (int): Index of the sample to retrieve.

        Returns:
            image (Tensor): Transformed image tensor.
            label (Tensor): Transformed label tensor.
        """
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        # Read the image
        image = cv2.imread(image_path)
        image = image / 255.0  # Normalize image to [0, 1]

        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)

        # Process label to have shape (H, W, NC) where NC = number of classes
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # Read the annotation JSON file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # Iterate over annotations to generate masks for each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # Create polygon mask for the class
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        # Apply transformations if specified
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        # Convert to PyTorch format (channel first)
        image = image.transpose(2, 0, 1)  # Move channel dimension to the first
        label = label.transpose(2, 0, 1)

        # Convert to tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label
