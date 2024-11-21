import os


WEBHOOK_URL = 'https://discord.com/api/webhooks/1306529597055041562/DUG0omhuBla0YM6SVqVgcSWgRBP2D0WZ5_xJt9aNvLL2QJpIHicb4tRupbvWYRLRtEgN'


YOLO_MODEL_PATH="/data/ephemeral/home/MCG/YOLO_Detection_Model/best.pt"

SAVE_VISUALIZE_TRAIN_DATA_PATH="/data/ephemeral/home/MCG/YOLO_Detection_Model/crop_train_Image"
VISUALIZE_TRAIN_DATA=True

IMAGE_ROOT = "/data/ephemeral/home/MCG/data/train/DCM"
LABEL_ROOT = "/data/ephemeral/home/MCG/data/train/outputs_json"

'''CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
'''

CLASSES = [
    'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform',
]


CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


YOLO_NAMES = {0: "finger", 1: "radius_ulna", 2: "others"}  # YOLO 클래스 인덱스 매핑
YOLO_SELECT_CLASS="others"


RANDOM_SEED = 21

# 적절하게 조절
NUM_EPOCHS =30
VAL_EVERY = 1

BATCH_SIZE = 1
IMSIZE=480

LR = 0.0001
MILESTONES=[7,16,23,27]
GAMMA=0.3


SAVED_DIR = "/data/ephemeral/home/MCG/UNetRefactored/Creadted_model/"
MODELNAME="othersCrop.pt"
if not os.path.isdir(SAVED_DIR):
    os.mkdir(SAVED_DIR)
    


INFERENCE_MODEL_NAME="othersCrop.pt"

TEST_IMAGE_ROOT="/data/ephemeral/home/MCG/data/test/DCM"

CSVDIR="/data/ephemeral/home/MCG/UNetRefactored/CSV"
CSVNAME="othersCrop"