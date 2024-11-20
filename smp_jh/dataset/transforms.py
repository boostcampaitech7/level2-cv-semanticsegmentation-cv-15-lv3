import albumentations as A

class Transforms:
    @staticmethod
    def get_train_transform():
        return A.Compose([
            A.Resize(512, 512),
            # TODO: Add more augmentations later
        ])

    @staticmethod
    def get_valid_transform():
        return A.Compose([
            A.Resize(512, 512),
        ])

    @staticmethod
    def get_test_transform():
        return A.Compose([
            A.Resize(512, 512),
        ])