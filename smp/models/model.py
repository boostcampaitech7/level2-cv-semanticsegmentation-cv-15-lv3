import segmentation_models_pytorch as smp
from config.config import Config

def get_model(num_classes=29):
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
    return model_fn(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=num_classes,
    )
