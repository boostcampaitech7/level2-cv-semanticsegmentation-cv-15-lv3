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

    # 수정 전 시각화
    print("\nSaving visualization before cleaning...")
    os.makedirs("visualization/before", exist_ok=True)
    
    print("\nChecking train dataset annotations before cleaning...")
    train_dataset.visualize_annotation("ID058", "image1661392103627.png", save_dir="visualization/before/train")
    train_dataset.visualize_annotation("ID363", "image1664935962797.png", save_dir="visualization/before/train")
    
    print("\nChecking validation dataset annotations before cleaning...")
    valid_dataset.visualize_annotation("ID058", "image1661392103627.png", save_dir="visualization/before/valid")
    valid_dataset.visualize_annotation("ID363", "image1664935962797.png", save_dir="visualization/before/valid")
    
    # 클렌징 작업 수행
    print("\nPerforming dataset cleaning...")
    
    # Train 데이터셋 클렌징
    print("\nCleaning train dataset...")
    train_dataset.fix_specific_annotation(
        id_folder="ID058",
        image_name="image1661392103627.png"
    )
    train_dataset.remove_specific_mask(
        id_folder="ID363",
        image_name="image1664935962797.png",
        label_to_remove="finger-14"
    )
    
    # Validation 데이터셋 클렌징
    print("\nCleaning validation dataset...")
    valid_dataset.fix_specific_annotation(
        id_folder="ID058",
        image_name="image1661392103627.png"
    )
    valid_dataset.remove_specific_mask(
        id_folder="ID363",
        image_name="image1664935962797.png",
        label_to_remove="finger-14"
    )
    
    # 수정 후 시각화
    print("\nSaving visualization after cleaning...")
    os.makedirs("visualization/after", exist_ok=True)
    
    print("\nChecking train dataset annotations after cleaning...")
    train_dataset.visualize_annotation("ID058", "image1661392103627.png", save_dir="visualization/after/train")
    train_dataset.visualize_annotation("ID363", "image1664935962797.png", save_dir="visualization/after/train")
    
    print("\nChecking validation dataset annotations after cleaning...")
    valid_dataset.visualize_annotation("ID058", "image1661392103627.png", save_dir="visualization/after/valid")
    valid_dataset.visualize_annotation("ID363", "image1664935962797.png", save_dir="visualization/after/valid")
    
    # 클렌징 결과 확인
    print("\nVerifying cleaned annotations...")
    print("\nTrain dataset:")
    train_dataset.clean_image_annotations("ID058", "image1661392103627.png")
    train_dataset.clean_image_annotations("ID363", "image1664935962797.png")
    
    print("\nValidation dataset:")
    valid_dataset.clean_image_annotations("ID058", "image1661392103627.png")
    valid_dataset.clean_image_annotations("ID363", "image1664935962797.png")
    
    print("\nData cleansing completed!")
    print("Please check the visualization folders to verify the changes.")

if __name__ == "__main__":
    clean_dataset()