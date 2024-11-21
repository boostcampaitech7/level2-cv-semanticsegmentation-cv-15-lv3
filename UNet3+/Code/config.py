import os

IMAGE_ROOT = "../../data/train/DCM"
LABEL_ROOT = "../../data/train/outputs_json"
CLASSES = [
    'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform'
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

BATCH_SIZE = 1
RANDOM_SEED = 21

# 적절하게 조절
NUM_EPOCHS =30
VAL_EVERY = 2
IMSIZE=1024
LR = 0.0001
MILESTONES=[6,16,23,27]
GAMMA=0.3

SAVED_DIR = "./Created_model/Unet3+"
MODELNAME="best_aug.pt"
    
INFERENCE_MODEL_NAME="best_CenterCrop.pt"

TEST_IMAGE_ROOT="../../data/test/DCM"

CSVDIR="./CSV"
CSVNAME="New_mmm"