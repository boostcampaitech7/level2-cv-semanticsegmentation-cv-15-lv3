import os
from dataset.data_loader import get_image_label_paths

TRAIN_IMAGE_ROOT = "C:\\uddaniiii\\boostcamp\\project\\level2\\segmentation\\data\\train\\DCM"
TRAIN_LABEL_ROOT = "C:\\uddaniiii\\boostcamp\\project\\level2\\segmentation\\data\\train\\outputs_json"

TEST_IMAGE_ROOT = "C:\\uddaniiii\\boostcamp\\project\\level2\\segmentation\\data\\test\\DCM"

train_pngs, train_jsons = get_image_label_paths(TRAIN_IMAGE_ROOT, TRAIN_LABEL_ROOT)
test_pngs, test_jsons = get_image_label_paths(TEST_IMAGE_ROOT)

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 2
LR = 1e-4
RANDOM_SEED = 21
WEIGHT_DECAY = 1e-6

NUM_EPOCHS = 50

SAVED_DIR = "./checkpoints"

ENCODER_NAME = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"
NUM_WORKERS = 0

if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)