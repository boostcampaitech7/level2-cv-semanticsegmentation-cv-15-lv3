import torch
import albumentations as A
import cv2
import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from dataset import XRayInferenceDataset
from config import IMAGE_ROOT, IND2CLASS, MODEL_NAME, CLASSES
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from utils import encode_mask_to_rle, decode_rle_to_mask

def test(model: torch.nn.Module) -> None:
    """
    Run inference on test dataset and save results to CSV.
    
    Args:
        model: PyTorch model for inference
    """
    # Setup dataset and dataloader
    tf = A.Resize(512, 512)
    test_dataset = XRayInferenceDataset(transforms=tf)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,       
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    # Run inference
    rles, filename_and_class = inference(model, test_loader)
    
    # Process predictions
    preds = [
        decode_rle_to_mask(rle, height=2048, width=2048)
        for rle in rles[:len(CLASSES)]
    ]
    preds = np.stack(preds, 0)
    
    # Prepare output dataframe
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    df.to_csv(f"{MODEL_NAME}.csv", index=False)

def inference(model: torch.nn.Module, data_loader: DataLoader, thr: float = 0.5) -> tuple[list, list]:
    """
    Run inference on given data loader.
    
    Args:
        model: PyTorch model for inference
        data_loader: DataLoader containing test data
        thr: Threshold for binary segmentation
        
    Returns:
        tuple containing:
            - list of RLE encoded masks
            - list of filename and class combinations
    """
    model = model.cuda()
    model.eval()
    
    rles = []
    filename_and_class = []
    
    with torch.no_grad():
        for images, image_names in tqdm(data_loader, total=len(data_loader)):
            images = images.cuda()
            outputs = model(images)
            
            # Restore original size and apply sigmoid
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            # Process each image in batch
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
    
    return rles, filename_and_class

