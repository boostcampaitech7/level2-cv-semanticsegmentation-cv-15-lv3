import os
import time
import datetime
import random
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from datetime import timedelta
from config.config import Config
from dataset.dataset import XRayDataset, StratifiedXRayDataset
from models.model import get_model
from utils.metrics import dice_coef
from dataset.transforms import Transforms
from models.loss import Loss
from models.scheduler import Scheduler
from utils.wandb import init_wandb
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def validation(epoch, model, data_loader, criterion, device, threshold=0.5):
    print(f"\nValidation Epoch {epoch}")
    val_start = time.time()
    model.eval()
    
    total_loss = 0
    dices = []
    first_batch_saved = False  # Flag to ensure the first batch visualization is saved

    with torch.no_grad():
        for step, (images, masks) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > threshold)
            dice = dice_coef(outputs, masks)
            dices.append(dice.detach().cpu())

            if not first_batch_saved:
                save_dir = f"validation_visualizations/epoch_{epoch}_batch_{step}"
                os.makedirs(save_dir, exist_ok=True)

                images_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # (B, H, W, C)
                for idx in range(images.size(0)):
                    plt.imshow(images_np[idx].squeeze(), cmap="gray")
                    plt.title("Input Image")
                    plt.axis("off")
                    plt.savefig(os.path.join(save_dir, f"image_{idx}.png"))
                    plt.close()

                print(f"First batch visualizations saved at {save_dir}")
                first_batch_saved = True
                break  # Save only the first batch and stop for visualization

    val_time = time.time() - val_start

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [f"{c:<12}: {d.item():.4f}" for c, d in zip(Config.CLASSES, dices_per_class)]
    print("\n".join(dice_str))

    avg_dice = torch.mean(dices_per_class).item()
    avg_loss = total_loss / len(data_loader)

    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Time: {datetime.timedelta(seconds=val_time)}\n")

    class_dice_dict = {f"{c}'s dice score": d.item() for c, d in zip(Config.CLASSES, dices_per_class)}
    return avg_dice, class_dice_dict, avg_loss


def train():
    set_seed(Config.RANDOM_SEED)

    init_wandb()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()

    model = get_model(num_classes=len(Config.CLASSES)).to(device)
    criterion = Loss.get_criterion(Config.LOSS_TYPE)
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=1e-6
    )
    scheduler = Scheduler.get_scheduler(
        scheduler_type=Config.SCHEDULER_TYPE,
        optimizer=optimizer,
        min_lr=Config.MIN_LR
    )

    train_dataset = StratifiedXRayDataset(
        image_root=Config.TRAIN_IMAGE_ROOT,
        label_root=Config.TRAIN_LABEL_ROOT,
        is_train=True,
        transforms=Transforms.get_train_transform(),
        meta_path=Config.META_PATH
    )

    valid_dataset = StratifiedXRayDataset(
        image_root=Config.TRAIN_IMAGE_ROOT,
        label_root=Config.TRAIN_LABEL_ROOT,
        is_train=False,
        transforms=Transforms.get_valid_transform(),
        meta_path=Config.META_PATH
    )

    print("\nDataset Statistics:")
    train_dataset.print_dataset_stats()
    valid_dataset.print_dataset_stats()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    patience = 10
    counter = 0
    best_dice = 0.0
    global_step = 0

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0

        for step, (images, masks) in enumerate(train_loader):
            break  # For debugging or specific use cases
            images = images.to(device)
            masks = masks.to(device)

            with autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(), 4)}'
                )
                wandb.log({
                    "Step Loss": loss.item(),
                    "Learning Rate": optimizer.param_groups[0]['lr']
                }, step=global_step)
                global_step += 1

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_epoch_loss:.4f} || Elapsed Time: {timedelta(seconds=epoch_time)}")

        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": avg_epoch_loss,
        }, step=global_step)

        if (epoch + 1) % Config.VAL_EVERY == 0:
            dice, dice_dict, val_loss = validation(epoch + 1, model, valid_loader, criterion, device)

            wandb.log({
                "Validation Loss": val_loss,
                "Average Dice Score": dice,
                **dice_dict,
            }, step=global_step)

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                best_dice = dice
                torch.save(model, os.path.join(Config.SAVED_DIR, "best_model.pt"))
                wandb.log({
                    "Best Dice Score": dice,
                    "Best Model Epoch": epoch + 1
                }, step=global_step)
                counter = 0
            else:
                counter += 1
                print(f"Early Stopping Counter: {counter} out of {patience}")
                if counter >= patience:
                    print(f"Early Stopping triggered! Best Dice: {best_dice:.4f}")
                    wandb.log({
                        "Early Stopping": epoch + 1,
                        "Final Best Dice": best_dice
                    }, step=global_step)
                    break

    wandb.finish()


if __name__ == "__main__":
    train()