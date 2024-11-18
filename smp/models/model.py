import segmentation_models_pytorch as smp

def get_model(num_classes=29):
    return smp.UnetPlusPlus(
        encoder_name="tu-hrnet_w64", # Timm encoder 사용시 이름 앞에 tu- 붙임
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )