import segmentation_models_pytorch as smp
from config import CLASSES

def get_smp_model(model_type="unetplusplus",
                  encoder_name="resnet50",
                  encoder_weights="imagenet",
                  in_channels=3,
                  classes=len(CLASSES)):
    """
    SMP 모델을 생성하고 반환하는 함수
    """
    if model_type == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif model_type == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    elif model_type == "manet":
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model