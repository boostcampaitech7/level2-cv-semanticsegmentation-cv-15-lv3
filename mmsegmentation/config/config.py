class Config:
    IMAGE_ROOT = "/data/ephemeral/home/jaehuni/level2-cv-semanticsegmentation-cv-15-lv3/data/train/DCM"
    LABEL_ROOT = "/data/ephemeral/home/jaehuni/level2-cv-semanticsegmentation-cv-15-lv3/data/train/outputs_json"

    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]

# 클래스 밖에서 직접 정의하여 export
CLASSES = Config.CLASSES
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}