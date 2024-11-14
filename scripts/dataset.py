import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import IMAGE_ROOT

class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        self.filenames = np.array(sorted(self._get_png_files()))
        self.transforms = transforms

    def _get_png_files(self):
        return {
            os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
            for root, _dirs, files in os.walk(IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        image = cv2.imread(image_path) / 255.0

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        image = image.transpose(2, 0, 1)  # make channel first
        image = torch.from_numpy(image).float()
        return image, image_name
