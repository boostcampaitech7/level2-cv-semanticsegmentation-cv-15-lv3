import albumentations as A
from config.config import Config

class Transforms:
    @staticmethod
    def get_train_transform():
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.RandomGamma(gamma_limit=(80, 200), p=0.3),
            A.Rotate(limit=10, p=0.8),  # Random Rotation (-12 ~ 12도, 70% 확률)
            A.HorizontalFlip(p=1),  # Horizontal Flip (항상 적용)
            A.RandomBrightnessContrast(
                brightness_limit=0.24,  # 밝기 조정 범위: ±20%
                contrast_limit=0.24,  # 대비 조정 범위: ±20%
                brightness_by_max=False,  # 정규화된 값 기준으로 밝기 조정
                p=0.8  # 50% 확률로 적용
             ),
        ])
    
    @staticmethod
    def get_train_ori():
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.RandomGamma(gamma_limit=(80, 200), p=0.3),
        ])
    
    @staticmethod
    def get_train_ori():
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.RandomGamma(gamma_limit=(80, 200), p=0.3)  # 감마 보정 추가
        ])

    @staticmethod
    def get_valid_transform():
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        ])

    @staticmethod
    def get_test_transform():
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        ])
