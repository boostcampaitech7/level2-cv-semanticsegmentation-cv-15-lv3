import albumentations as A
from config.config import Config

class Transforms:
    @staticmethod
    def get_train_transform():
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            # TODO: Add more augmentations later
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
