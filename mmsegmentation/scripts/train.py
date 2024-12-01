import os
import sys

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmengine.config import Config
from mmengine.runner import Runner

# custom modules import 추가
import custom_xray.models.segmentors  # EncoderDecoderWithoutArgmax
import custom_xray.models.heads       # SegformerHeadWithoutAccuracy
import custom_xray.datasets.xray_dataset  # XRayDataset
import custom_xray.transforms.loading  # LoadXRayAnnotations, TransposeAnnotations
import utils.metrics  # DiceMetric

def main():
    cfg = Config.fromfile("../custom_xray/models/segformer.py")
    cfg.launcher = "none"
    cfg.work_dir = "work_dirs/segformer"
    cfg.resume = False

    # GPU 설정 추가
    cfg.device = 'cuda'

    # 학습 시작
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == "__main__":
    main()