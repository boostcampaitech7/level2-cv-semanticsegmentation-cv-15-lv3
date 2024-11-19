import albumentations as A
from DataSet.Dataset import XRayInferenceDataset
from torch.utils.data import DataLoader
from config import SAVED_DIR, INFERENCE_MODEL_NAME, IMSIZE, CSVDIR,CSVNAME
import os
import torch
from Infrence import test
import pandas as pd

model = torch.load(os.path.join(SAVED_DIR, INFERENCE_MODEL_NAME))

tf = A.Resize(IMSIZE, IMSIZE)
test_dataset = XRayInferenceDataset(transforms=tf)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

rles, filename_and_class = test(model, test_loader)

classes, filename = zip(*[x.split("_") for x in filename_and_class])

image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

df.to_csv(os.path.join(CSVDIR, CSVNAME),index=False)