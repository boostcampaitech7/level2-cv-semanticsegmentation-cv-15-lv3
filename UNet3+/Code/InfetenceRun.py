import os
import albumentations as A
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm.auto import tqdm
import torch.nn.functional as F
import importlib

import config


def get_model_class(model_name):
    module_name, class_name = model_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=config.TEST_IMAGE_ROOT)
            for root, _dirs, files in os.walk(config.TEST_IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        self.filenames = np.array(sorted(pngs))
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(config.TEST_IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tensor will be done later
        image = image.transpose(2, 0, 1)  # make channel first
        image = torch.from_numpy(image).float()

        return image, image_name


def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(config.CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            # Restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{config.IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


if __name__ == "__main__":
    # 동적으로 모델 로드
    model_name = f"Model.{config.INFERENCE_MODEL_NAME}"
    ModelClass = get_model_class(model_name)
    model = ModelClass()
    model.load_state_dict(torch.load(os.path.join(config.SAVED_DIR, config.INFERENCE_MODEL_NAME)))

    tf = A.Resize(config.IMSIZE, config.IMSIZE)
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

    df.to_csv(os.path.join(config.CSVDIR, config.CSVNAME), index=False)
