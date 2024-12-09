import os
import importlib
import albumentations as A
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm.auto import tqdm
from ultralytics import YOLO
from torch.multiprocessing import set_start_method
import torch.nn.functional as F

import config
from DataSet.YoloInferenceDataset import XRayInferenceDataset


def keep_largest_connected_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_model_class(model_name):
    module_name, class_name = model_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        CLASS_COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 165, 0)
        ]

        for step, (images, image_names, crop_boxes) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            outputs = model(images)

            for output, image, image_name, crop_box in zip(outputs, images, image_names, crop_boxes):
                start_x, start_y, end_x, end_y = crop_box
                output = torch.sigmoid(output).detach().cpu().numpy()
                crop_width, crop_height = end_x - start_x, end_y - start_y
                output_height, output_width = output.shape[-2:]

                if (output_width != crop_width) or (output_height != crop_height):
                    raise ValueError("Output size does not match crop size!")

                image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                overlay_mask = np.zeros_like(image, dtype=np.uint8)

                for c, segm in enumerate(output):
                    binary_mask = (segm > thr).astype(np.uint8)
                    color = CLASS_COLORS[c]
                    color_mask = np.zeros_like(overlay_mask, dtype=np.uint8)
                    for i in range(3):
                        color_mask[:, :, i] = binary_mask * color[i]
                    overlay_mask = cv2.add(overlay_mask, color_mask)

                blended_image = cv2.addWeighted(image, 0.7, overlay_mask, 0.3, 0)
                clean_image_name = os.path.basename(image_name)
                save_path = f"/data/ephemeral/home/MCG/YOLO_Detection_Model/test_last/{clean_image_name}"
                cv2.imwrite(save_path, cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))

                for c, segm in enumerate(output):
                    binary_mask = (segm > thr).astype(np.uint8)
                    largest_component = keep_largest_connected_component(binary_mask)
                    full_size_mask = np.zeros((2048, 2048), dtype=np.uint8)
                    full_size_mask[start_y:end_y, start_x:end_x] = largest_component
                    rle = encode_mask_to_rle(full_size_mask)
                    rles.append(rle)
                    filename_and_class.append(f"{config.IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


if __name__ == "__main__":
    set_start_method("spawn", force=True)

    # 동적으로 모델 로드
    model_name = f"Model.{config.INFERENCE_MODEL_NAME}"
    ModelClass = get_model_class(model_name)
    model = ModelClass()
    model.load_state_dict(torch.load(os.path.join(config.SAVED_DIR, config.INFERENCE_MODEL_NAME)))

    yolo_model = YOLO("/data/ephemeral/home/MCG/YOLO_Detection_Model/best.pt")

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=config.TEST_IMAGE_ROOT)
        for root, _, files in os.walk(config.TEST_IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    test_dataset = XRayInferenceDataset(
        filenames=pngs,
        yolo_model=yolo_model,
        save_dir=config.SAVE_VISUALIZE_TRAIN_DATA_PATH,
        draw_enabled=True
    )

    def custom_collate_fn(batch):
        images, image_names, crop_boxes = zip(*batch)
        return (
            torch.stack(images),
            list(image_names),
            list(crop_boxes)
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    rles, filename_and_class = test(model, test_loader, thr=0.5)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(os.path.join(config.CSVDIR, config.CSVNAME), index=False)
