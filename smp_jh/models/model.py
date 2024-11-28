import segmentation_models_pytorch as smp
from config.config import Config

import torch
import torch.nn as nn

class PreprocessingLayer(nn.Module):
    def __init__(self):
        super(PreprocessingLayer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),  # 2048 → 1024
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),  # 1024 → 512
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv_layers(x)


# 커스텀 모델 정의 (PreprocessingLayer + Segmentation Model 결합)
class CombinedModel(nn.Module):
    def __init__(self, preprocess_layer, segmentation_model):
        super(CombinedModel, self).__init__()
        self.preprocess = preprocess_layer
        self.segmentation = segmentation_model

    def forward(self, x):
        x = self.preprocess(x)  # 전처리 레이어를 통해 크기 축소
        x = self.segmentation(x)  # Segmentation 모델에 전달
        return x

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

    segmentation_model = model_fn(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=num_classes,
    )

    preprocess_layer = PreprocessingLayer()
    combined_model = CombinedModel(preprocess_layer, segmentation_model)

    return combined_model

    # return model_fn(
    #     encoder_name=Config.ENCODER_NAME,
    #     encoder_weights=Config.ENCODER_WEIGHTS,
    #     in_channels=3,
    #     classes=num_classes,
    # )


# # Segmentation 모델 생성 함수
# def get_model(num_classes=29):
#     # 지원하는 모델 리스트
#     MODELS = {
#         'Unet': smp.Unet,
#         'UnetPlusPlus': smp.UnetPlusPlus,
#         'FPN': smp.FPN,
#         'PSPNet': smp.PSPNet,
#         'DeepLabV3': smp.DeepLabV3,
#         'DeepLabV3Plus': smp.DeepLabV3Plus,
#         'Linknet': smp.Linknet,
#         'MAnet': smp.MAnet,
#         'PAN': smp.PAN,
#         'UPerNet': smp.UPerNet,
#     }

#     # Config에서 모델 선택
#     model_fn = MODELS[Config.MODEL_ARCHITECTURE]

#     # Segmentation 모델 생성
#     segmentation_model = model_fn(
#         encoder_name=Config.ENCODER_NAME,  # Encoder 이름 (예: resnet34)
#         encoder_weights=Config.ENCODER_WEIGHTS,  # Pretrained 가중치 (예: 'imagenet')
#         in_channels=3,  # 입력 채널 (RGB)
#         classes=num_classes,  # 출력 클래스 수
#     )

#     # 전처리 레이어와 결합
#     preprocess_layer = PreprocessingLayer()
#     combined_model = CombinedModel(preprocess_layer, segmentation_model)

#     return combined_model