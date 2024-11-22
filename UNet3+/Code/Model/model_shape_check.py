from torchvision.models import convnext_large
import torch

# ConvNeXt Large 모델 로드
#model = convnext_large(pretrained=True)

# 모델 구조 출력
#print(model)

ce_loss = torch.log(torch.tensor(1e-6))
print(ce_loss)