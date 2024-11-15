# Constants
IMAGE_ROOT = "../data/train/DCM"
LABEL_ROOT = "../data/train/outputs_json"
TEST_IMAGE_ROOT = "../data/test/DCM"

MODE = "train"

BATCH_SIZE = 1
LR = 1e-3
NUM_EPOCHS = 20
VAL_EVERY = 1
RANDOM_SEED = 21

DISCORD_ALERT = True
HEIGHT = 512
WIDTH = 512

MODEL_NAME = "ducknet"
SAVED_DIR = f"../checkpoints/{MODEL_NAME}"
# Discord Webhook URL
SERVER_ID = "3"
WEBHOOK_URL = 'https://discord.com/api/webhooks/1305343891964428318/G85AIWjdio2VBY7V-egcaI-qJDOOcRAAVrThsUh6yYmKMdKT5Ff4HNkMkk8gWkkCNdWV'

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

