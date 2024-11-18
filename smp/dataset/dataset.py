import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from config.config import Config

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root=None, is_train=True, transforms=None):
        self.is_train = is_train
        self.transforms = transforms
        self.CLASS2IND = Config.CLASS2IND
        
        # Get all PNG files
        self.image_root = image_root
        self.label_root = label_root
        
        self.pngs = self._get_pngs()
        if is_train:
            self.jsons = self._get_jsons()
            
            # Verify matching between pngs and jsons
            jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in self.jsons}
            pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in self.pngs}
            
            # Check if all files match
            assert len(jsons_fn_prefix - pngs_fn_prefix) == 0, "Some JSON files don't have matching PNGs"
            assert len(pngs_fn_prefix - jsons_fn_prefix) == 0, "Some PNG files don't have matching JSONs"
            
            self.filenames, self.labelnames = self._split_dataset()
        else:
            self.filenames = sorted(self.pngs)
            
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
        
    def _split_dataset(self):
        _filenames = np.array(self.pngs)
        _labelnames = np.array(self.jsons)
        
        # Split train-valid using GroupKFold
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for _ in _filenames]
        
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                if i == 0:  # Use fold 0 as validation
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                break
                
        return filenames, labelnames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        if self.is_train:
            label_name = self.labelnames[item]
            label_path = os.path.join(self.label_root, label_name)
            
            # Create label with shape (H, W, NC)
            label_shape = tuple(image.shape[:2]) + (29,)  # 29 classes
            label = np.zeros(label_shape, dtype=np.uint8)

            with open(label_path, "r") as f:
                annotations = json.load(f)
            annotations = annotations["annotations"]

            # Process each class
            for ann in annotations:
                c = ann["label"]
                class_ind = self.CLASS2IND[c]
                points = np.array(ann["points"])

                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_ind] = class_label

            if self.transforms is not None:
                inputs = {"image": image, "mask": label}
                result = self.transforms(**inputs)
                image = result["image"]
                label = result["mask"]

            # Convert to tensor format
            image = image.transpose(2, 0, 1)
            label = label.transpose(2, 0, 1)

            return torch.from_numpy(image).float(), torch.from_numpy(label).float()
        else:
            if self.transforms is not None:
                inputs = {"image": image}
                result = self.transforms(**inputs)
                image = result["image"]

            image = image.transpose(2, 0, 1)
            return torch.from_numpy(image).float(), image_name

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