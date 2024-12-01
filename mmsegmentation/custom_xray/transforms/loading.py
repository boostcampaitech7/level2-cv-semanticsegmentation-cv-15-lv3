from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import numpy as np
import cv2
import json
from config.config import Config, CLASS2IND

@TRANSFORMS.register_module()
class LoadXRayAnnotations(BaseTransform):
    def transform(self, result):
        label_path = result["seg_map_path"]
        image_size = (2048, 2048)
        label_shape = image_size + (len(Config.CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            class_label = np.zeros(image_size, dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        result["gt_seg_map"] = label
        return result

@TRANSFORMS.register_module()
class TransposeAnnotations(BaseTransform):
    def transform(self, result):
        result["gt_seg_map"] = np.transpose(result["gt_seg_map"], (2, 0, 1))
        return result