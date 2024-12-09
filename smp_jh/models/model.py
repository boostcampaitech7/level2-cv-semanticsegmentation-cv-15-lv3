import segmentation_models_pytorch as smp
from config.config import Config
from .encoders.register import register_swin_encoder

def get_model(num_classes=29):
    # Register custom encoders
    register_swin_encoder()

    MODELS = {
        'Unet': smp.Unet,
        'UnetPlusPlus': smp.UnetPlusPlus,
        'FPN': smp.FPN,
        'PSPNet': smp.PSPNet,
        'DeepLabV3': smp.DeepLabV3,
        'DeepLabV3Plus': smp.DeepLabV3Plus,
        'Linknet': smp.Linknet,
        'MAnet': smp.MAnet,
        'PAN': smp.PAN,
        'UPerNet': smp.UPerNet,
    }
    
    model_fn = MODELS[Config.MODEL_ARCHITECTURE]

    decoder_channels = (256, 128, 64, 32, 16)

    if Config.ENCODER_NAME == "swin":
        return model_fn(
            encoder_name="swin_encoder",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=True,
            upsampling=2,
        )
    
    return model_fn(
        encoder_name="swin_encoder" if Config.ENCODER_NAME == "swin" else Config.ENCODER_NAME,
        encoder_weights=None if Config.ENCODER_NAME == "swin" else Config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=num_classes,
    )
