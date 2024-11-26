from pathlib import Path

class Config:
    # Data
    TRAIN_IMAGE_ROOT = "../data/train/DCM"
    TRAIN_LABEL_ROOT = "../data/train/outputs_json"
    TEST_IMAGE_ROOT = "../data/test/DCM"
    META_PATH = "../data/meta_data.xlsx"
    
    # Model
    MODEL_ARCHITECTURE = 'UnetPlusPlus' # [Unet, UnetPlusPlus, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, LinkNet, MAnet, PAN, UPerNet]
    ENCODER_NAME = 'tu-hrnet_w64' # encoder 이름 Timm encoder 사용시 이름 앞에 tu- 붙임
    ENCODER_WEIGHTS = 'imagenet' # pretrained weights

    TRAIN_BATCH_SIZE = 2
    VAL_BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    VAL_EVERY = 1
    RANDOM_SEED = 21

    IMG_SIZE = 512

    # Loss
    LOSS_TYPE = "bce" # [ "bce", "dice", "focal" ]

    # Scheduler
    # train.py 에서 주석을 풀어야 사용 가능 아직 적용 x
    SCHEDULER_TYPE = "reduce" # [ "reduce", "step", "cosine" ]
    MIN_LR = 1e-6
    
    WANDB = {
        "api_key": "your_api_key",
        "project_name": "Hand_bone_segmentation",
        "experiment_detail": f"{MODEL_ARCHITECTURE}_{ENCODER_NAME}_batch{TRAIN_BATCH_SIZE}_{NUM_EPOCHS}ep",
        "model_name": MODEL_ARCHITECTURE,
    }

    # Paths
    SAVED_DIR = Path("checkpoints")
    SAVED_DIR.mkdir(exist_ok=True)
    
    # Classes
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