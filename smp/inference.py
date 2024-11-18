import os
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from config.config import Config
from dataset.dataset import XRayDataset
from utils.rle import encode_mask_to_rle
from dataset.transforms import Transforms  # Transforms 클래스 import

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            outputs = model(images)['out']
            
            # Resize to original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{Config.IND2CLASS[c]}_{image_name}")
    
    return rles, filename_and_class

def main():
    # 데이터셋 준비
    test_dataset = XRayDataset(
        image_root=Config.TEST_IMAGE_ROOT,
        is_train=False,
        transforms=Transforms.get_test_transform()  # Transforms 클래스 사용
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    # 모델 로드
    model = torch.load(os.path.join(Config.SAVED_DIR, "best_model.pt"))
    
    # 추론
    rles, filename_and_class = test(model, test_loader)
    
    # 결과를 DataFrame으로 변환
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    # CSV 저장
    df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()