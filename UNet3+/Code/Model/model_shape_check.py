import torch
from torchvision.models import resnet152
from torchsummary import summary

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
