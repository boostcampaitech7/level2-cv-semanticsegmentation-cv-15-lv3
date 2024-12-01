'''import torch
from torchvision.models import resnet152
from torchsummary import summary
from models.cls_hrnet import get_cls_net

# ResNet152 모델 로드
model = resnet152(pretrained=True)

# 임의의 입력 생성 (배치 크기 1, 채널 수 3, 크기 224x224)
input_tensor = torch.randn(1, 3, 224, 224)

# 각 레이어를 통과하며 크기 추적
def print_layer_shapes(model, input_tensor):
    x = input_tensor
    print(f"{'Layer':<30}{'Input Shape':<30}{'Output Shape':<30}")
    print("=" * 90)
    for name, layer in model.named_children():
        x = layer(x)
        print(f"{name:<30}{str(tuple(input_tensor.shape)):<30}{str(tuple(x.shape)):<30}")
        input_tensor = x

print_layer_shapes(model, input_tensor)
'''

from Model.HRnetModel import UNet3PlusHRNet
import torch

# YAML 파일 경로
yaml_file = "/data/ephemeral/home/MCG/level2-cv-semanticsegmentation-cv-15-lv3/UNet3+/Code/HRNet/experiments/w48.yaml"
pretrained_weights = "/data/ephemeral/home/MCG/hrnetv2_w48_imagenet_pretrained.pth"

# 모델 초기화
model = UNet3PlusHRNet(in_channels=3, n_classes=1, hrnet_config_file=yaml_file, pretrained_weights=pretrained_weights)

# 입력 텐서 생성
input_tensor = torch.randn(1, 3, 224, 224)  # Batch=1, Channels=3, Height=224, Width=224

# Forward Pass
h1 = model.stage1(input_tensor)
h2 = model.stage2(h1)
h3 = model.stage3(h2)
h4 = model.stage4(h3)
h5 = model.stage5(h4)

# 출력 Shape 확인
print("Stage 1 Output Shape:", h1.shape)
print("Stage 2 Output Shape:", h2.shape)
print("Stage 3 Output Shape:", h3.shape)
print("Stage 4 Output Shape:", h4.shape)
print("Stage 5 Output Shape:", h5.shape)