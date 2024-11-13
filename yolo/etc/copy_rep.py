import os
import shutil

# 원본 폴더 경로와 복사할 대상 폴더 경로
original_folder = '/data/ephemeral/home/dataset'
copied_folder = '/data/ephemeral/home/dataset_yolo'

# 원본 폴더 전체를 대상 폴더로 복사
if not os.path.exists(copied_folder):  # 대상 폴더가 없을 경우에만 복사
    shutil.copytree(original_folder, copied_folder)
    print(f"{original_folder}를 {copied_folder}로 성공적으로 복사했습니다.")

# 복사할 파일이 있는 원본 상위 폴더 경로들
base_folders = [
    '/data/ephemeral/home/dataset/train/DCM', 
    '/data/ephemeral/home/dataset/train/outputs_json'
]

# 파일을 복사할 목적지 상위 폴더 경로들
destination_folders = [
    '/data/ephemeral/home/dataset_yolo/train/images', 
    '/data/ephemeral/home/dataset_yolo/train/labels'
]

# 목적지 폴더 생성 (이미 존재하면 무시)
for destination_folder in destination_folders:
    os.makedirs(destination_folder, exist_ok=True)

# 각 원본 폴더와 목적지 폴더 쌍마다 반복
for base_folder, destination_folder in zip(base_folders, destination_folders):
    # 새로운 파일 이름을 위한 초기 카운터
    file_counter = 1
    
    # 원본 상위 폴더 내 모든 하위 폴더 검색
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        # 폴더인지 확인
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                # 파일인지 확인
                if os.path.isfile(file_path):
                    # 새로운 파일 이름 생성 (image_1, image_2, ...)
                    new_file_name = f"image_{file_counter}{os.path.splitext(file_name)[1]}"
                    new_file_path = os.path.join(destination_folder, new_file_name)
                    # 파일 복사 및 이름 변경
                    shutil.copy2(file_path, new_file_path)
                    file_counter += 1  # 카운터 증가
