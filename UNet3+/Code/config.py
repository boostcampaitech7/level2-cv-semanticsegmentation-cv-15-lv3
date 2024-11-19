import os

IMAGE_ROOT = "/data/ephemeral/home/MCG/data/train/DCM"
LABEL_ROOT = "/data/ephemeral/home/MCG/data/train/outputs_json"
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

BATCH_SIZE = 2
RANDOM_SEED = 21

# 적절하게 조절
NUM_EPOCHS =30
VAL_EVERY = 1
IMSIZE=512
LR = 0.0001
MILESTONES=[6,16,23,27]
GAMMA=0.3


SAVED_DIR = "/data/ephemeral/home/MCG/UNetRefactored/Creadted_model/"
MODELNAME="best_NewModel.pt"
if not os.path.isdir(SAVED_DIR):
    os.mkdir(SAVED_DIR)
    


INFERENCE_MODEL_NAME="best_HybridLoss_ChangeSet.pt"

TEST_IMAGE_ROOT="/data/ephemeral/home/MCG/data/test/DCM"

CSVDIR=""
CSVNAME=""