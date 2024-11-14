import datetime
import os
import torch
import numpy as np
import random
import torch.nn.functional as F
import tqdm
from config import CLASSES, SAVED_DIR, WEBHOOK_URL
import requests

# Discord Webhook URL
webhook_url = WEBHOOK_URL
def send_discord_message(content):
    """Discord Webhook으로 메시지 전송"""
    data = {"content": content}
    response = requests.post(webhook_url, json=data)
    if response.status_code == 204:
        print("메시지가 성공적으로 전송되었습니다.")
    else:
        print("메시지 전송 실패:", response.status_code)
        

def dice_coef(y_true, y_pred):
    """
    Calculate the Dice coefficient (a measure of overlap) between true and predicted labels.

    Args:
        y_true (Tensor): Ground truth labels, shape [batch_size, num_classes, height, width]
        y_pred (Tensor): Predicted labels, shape [batch_size, num_classes, height, width]

    Returns:
        Tensor: Dice coefficient for each class in the batch, shape [batch_size]
    """
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


def save_model(model, file_name='best_model.pt'):
    """
    Save the trained model to a file.

    Args:
        model (torch.nn.Module): The model to be saved.
        output_path (str): Directory path where the model will be saved.
        file_name (str, optional): Name of the saved model file. Defaults to 'best_model.pt'.
    """
    if not os.path.isdir(SAVED_DIR):
        os.mkdir(SAVED_DIR)
        
    torch.save(model.state_dict(), f"{SAVED_DIR}/{file_name}")
    print(f"Model saved to {SAVED_DIR}/{file_name}")

def set_seed(seed=21):
    """
    Set the seed for reproducibility across various libraries (PyTorch, NumPy, and Python's random module).

    Args:
        seed (int, optional): The seed value to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set to {seed}")
           
def encode_mask_to_rle(mask: np.ndarray) -> str:
    """
    Encode binary mask to RLE format.
    
    Args:
        mask: Binary mask as numpy array (1 - mask, 0 - background)
        
    Returns:
        RLE encoded string
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    """
    Decode RLE string to binary mask.
    
    Args:
        rle: RLE encoded string
        height: Height of output mask
        width: Width of output mask
        
    Returns:
        Binary mask as numpy array
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)