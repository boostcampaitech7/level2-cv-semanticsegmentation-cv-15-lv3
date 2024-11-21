
# python native
import os

from sklearn.model_selection import GroupKFold
import albumentations as A
# torch
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from Model.FixedModel import UNet_3Plus_DeepSup
from DataSet.DataLoder import get_image_label_paths
from config import IMAGE_ROOT,LABEL_ROOT,BATCH_SIZE,IMSIZE,CLASSES,MILESTONES, GAMMA,LR, SAVED_DIR
from DataSet.Dataset import XRayDataset
from Loss.Loss import CombinedLoss
from Train import train
from Util.SetSeed import set_seed

set_seed()

if not os.path.isdir(SAVED_DIR):
    os.makedirs(SAVED_DIR)

pngs, jsons=get_image_label_paths(IMAGE_ROOT=IMAGE_ROOT,LABEL_ROOT=LABEL_ROOT)
#print(pngs, jsons)

# split train-valid
# 한 폴더 안에 한 인물의 양손에 대한 `.png` 파일이 존재하기 때문에
# 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
# 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
groups = [os.path.dirname(fname) for fname in pngs]

# dummy label
ys = [0 for fname in pngs]

# 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
# 5으로 설정하여 GroupKFold를 수행합니다.
gkf = GroupKFold(n_splits=5)

train_filenames = []
train_labelnames = []
valid_filenames = []
valid_labelnames = []
for i, (x, y) in enumerate(gkf.split(pngs, ys, groups)):
    # 0번을 validation dataset으로 사용합니다.
    if i == 0:
        valid_filenames += list(pngs[y])
        valid_labelnames += list(jsons[y])

    else:
        train_filenames += list(pngs[y])
        train_labelnames += list(jsons[y])

tf = A.Resize(IMSIZE,IMSIZE)
train_dataset = XRayDataset(train_filenames, train_labelnames, transforms=tf, is_train=True)
valid_dataset = XRayDataset(valid_filenames, valid_labelnames, transforms=tf, is_train=False)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=False
)


model = UNet_3Plus_DeepSup(n_classes=len(CLASSES))

# Loss function 정의
criterion = CombinedLoss(focal_weight=1, iou_weight=1, ms_ssim_weight=1, dice_weight=1)

# Optimizer 정의
optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)


train(model, train_loader, valid_loader, criterion, optimizer,scheduler)