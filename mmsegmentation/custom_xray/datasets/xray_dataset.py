from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import numpy as np
from sklearn.model_selection import GroupKFold
import os
from config.config import Config

IMAGE_ROOT = Config.IMAGE_ROOT
LABEL_ROOT = Config.LABEL_ROOT

@DATASETS.register_module()
class XRayDataset(BaseSegDataset):
    def __init__(self, is_train, **kwargs):
        self.is_train = is_train
        kwargs['data_root'] = '/'  # 기본값 설정
        super().__init__(**kwargs)

    def load_data_list(self):
        # 노트북과 동일한 방식으로 파일 탐색
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
            for root, _dirs, files in os.walk(IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }

        jsons = {
            os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
            for root, _dirs, files in os.walk(LABEL_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        
        if len(pngs) == 0:
            raise ValueError(f"No images found in {IMAGE_ROOT}")
            
        _filenames = np.array(list(pngs))
        _labelnames = np.array(list(jsons))

        # split train-valid
        groups = [os.path.dirname(fname) if os.path.dirname(fname) else 'root' for fname in _filenames]
        ys = [0] * len(_filenames)

        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                if i == 0:
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                break

        data_list = []
        for img_path, ann_path in zip(filenames, labelnames):
            data_info = dict(
                img_path=os.path.join(IMAGE_ROOT, img_path),
                seg_map_path=os.path.join(LABEL_ROOT, ann_path)
            )
            data_list.append(data_info)

        return data_list