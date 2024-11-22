from torchvision.models import convnext_large

# ConvNeXt Large 모델 로드
model = convnext_large(pretrained=True)

# 모델 구조 출력
print(model)