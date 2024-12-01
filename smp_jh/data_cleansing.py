import os
from config.config import Config
from dataset.dataset import StratifiedXRayDataset
from dataset.transforms import Transforms

def clean_dataset():
    """
    train과 validation 데이터셋 모두에 대해 클렌징 작업을 수행하는 메인 함수
    """
    print("Initializing datasets...")
    
    # Train 데이터셋 초기화
    train_dataset = StratifiedXRayDataset(
        image_root=Config.TRAIN_IMAGE_ROOT,
        label_root=Config.TRAIN_LABEL_ROOT,
        is_train=True,
        transforms=None,  # 클렌징 작업에는 transform 불필요
        meta_path=Config.META_PATH
    )
    
    # Validation 데이터셋 초기화
    valid_dataset = StratifiedXRayDataset(
        image_root=Config.TRAIN_IMAGE_ROOT,
        label_root=Config.TRAIN_LABEL_ROOT,
        is_train=False,
        transforms=None,
        meta_path=Config.META_PATH
    )
    
    # 클렌징 작업 수행
    print("\nPerforming dataset cleaning...")
    
    # Train 데이터셋 클렌징
    print("\nCleaning train dataset...")
    train_dataset.fix_specific_annotation(
        id_folder="ID058",
        image_name="image1661392103627.png"
    )
    
    # Validation 데이터셋 클렌징
    print("\nCleaning validation dataset...")
    valid_dataset.fix_specific_annotation(
        id_folder="ID058",
        image_name="image1661392103627.png"
    )
    
    # 클렌징 결과 확인
    print("\nVerifying cleaned annotations...")
    print("\nTrain dataset:")
    train_dataset.clean_image_annotations("ID058", "image1661392103627.png")
    
    print("\nValidation dataset:")
    valid_dataset.clean_image_annotations("ID058", "image1661392103627.png")
    
    print("\nData cleansing completed!")
    print("Please check the visualization folders to verify the changes.")

if __name__ == "__main__":
    clean_dataset()