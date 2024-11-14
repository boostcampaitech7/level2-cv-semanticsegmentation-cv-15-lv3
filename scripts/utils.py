import datetime
import torch
import numpy as np
import random
import torch.nn.functional as F
import tqdm

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

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


def save_model(model, output_path, file_name='best_model.pt'):
    """
    Save the trained model to a file.

    Args:
        model (torch.nn.Module): The model to be saved.
        output_path (str): Directory path where the model will be saved.
        file_name (str, optional): Name of the saved model file. Defaults to 'best_model.pt'.
    """
    torch.save(model.state_dict(), f"{output_path}/{file_name}")
    print(f"Model saved to {output_path}/{file_name}")


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

           
