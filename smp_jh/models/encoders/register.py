import segmentation_models_pytorch as smp
from .swin import SwinEncoder

def register_swin_encoder():
    smp.encoders.encoders["swin_encoder"] = {
        "encoder": SwinEncoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {}  # timm이 모든 파라미터를 처리하므로 비워둠
    }