import matplotlib.pyplot as plt
import numpy as np
from dataset import XRayDataset
import albumentations as A

# 시각화를 위한 팔레트 설정
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# 시각화 함수: label을 RGB로 변환
def label2rgb(label):
    image_size = label.shape[1:] + (3,)
    image = np.zeros(image_size, dtype=np.uint8)

    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]

    return image

# 데이터셋 로딩
tf = A.Resize(512, 512)
train_dataset = XRayDataset(is_train=True, transforms=tf)
valid_dataset = XRayDataset(is_train=False, transforms=tf)

# 샘플 이미지와 라벨 불러오기
image, label = train_dataset[0]

# 이미지와 라벨 시각화
fig, ax = plt.subplots(1, 2, figsize=(24, 12))
ax[0].imshow(image[0])  # 채널 차원 생략
ax[1].imshow(label2rgb(label))

plt.show()
