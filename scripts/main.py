from config import *
from dataset import XRayInferenceDataset
from model import UNetPlusPlus
from train import train
from inference import test
# from utils import encode_mask_to_rle, decode_rle_to_mask
# from visualize import visualize
# from notifications import send_discord_message
from discord_notifications import monitor_gpu, send_discord_message
import threading

def main():
    # Discord에 시작 알림 전송
    send_discord_message("🚀 [서버 3] 작업이 시작되었습니다.")

    # Initialize dataset, model, optimizer, etc.
    dataset = XRayInferenceDataset()
    model = UNetPlusPlus()
    # Define optimizer and loss function
    # Train the model
    # train(model, dataset)

    # Discord에 종료 알림 전송
    send_discord_message("✅ [서버 3] 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()