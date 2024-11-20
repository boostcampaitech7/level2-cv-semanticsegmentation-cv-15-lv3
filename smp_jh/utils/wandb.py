import wandb
from config.config import Config

def init_wandb():
    wandb.login(key=Config.WANDB['api_key'])
    wandb.init(
        project=Config.WANDB['project_name'],
        name=Config.WANDB['experiment_detail'],
        config={
            'model': Config.WANDB['model_name'],
            'batch_size': Config.TRAIN_BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'scheduler': Config.SCHEDULER_TYPE,
            'loss': Config.LOSS_TYPE,
            'epochs': Config.NUM_EPOCHS,
        }
    )