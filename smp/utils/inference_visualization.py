import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import decode_rle_to_mask
from config import IMAGE_ROOT, CLASSES

# RLE로부터 복원된 마스크와 이미지를 시각화하는 함수
def visualize_predictions(rles, filename_and_class):
    # 첫 번째 이미지 불러오기
    image = cv2.imread(os.path.join(IMAGE_ROOT, filename_and_class[0].split("_")[1]))

    preds = []
    for rle in rles[:len(CLASSES)]:
        # 각 RLE를 마스크로 복원
        pred = decode_rle_to_mask(rle, height=2048, width=2048)
        preds.append(pred)

    preds = np.stack(preds, 0)

    # 이미지와 마스크 시각화
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image)  # 원본 이미지 표시
    ax[1].imshow(label2rgb(preds))  # 예측된 마스크를 컬러로 시각화

    plt.show()

# 실행 예시
if __name__ == "__main__":
    visualize_predictions(rles, filename_and_class)
