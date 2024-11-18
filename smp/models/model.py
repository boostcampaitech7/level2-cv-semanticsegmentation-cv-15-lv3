import segmentation_models_pytorch as smp

def get_model(num_classes=29):
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )