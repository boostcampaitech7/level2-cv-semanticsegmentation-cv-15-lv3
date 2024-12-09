import os
import shutil
import json
from glob import glob
from natsort import natsorted

# 기본 설정
original_folder = '/data/ephemeral/home/dataset'
copied_folder = '/data/ephemeral/home/dataset_yolo'

# 1. 원본 폴더 전체 복사
if not os.path.exists(copied_folder):
    shutil.copytree(original_folder, copied_folder)
    print(f"{original_folder}를 {copied_folder}로 성공적으로 복사했습니다.")

# 2. 파일을 특정 폴더로 복사
base_folders = [
    '/data/ephemeral/home/dataset/train/DCM',
    '/data/ephemeral/home/dataset/train/outputs_json',
    '/data/ephemeral/home/dataset/test/DCM'
]

destination_folders = [
    '/data/ephemeral/home/dataset_yolo/images/train',
    '/data/ephemeral/home/dataset_yolo/labels/train',
    '/data/ephemeral/home/dataset_yolo/test'
]

# 목적지 폴더 생성
for destination_folder in destination_folders:
    os.makedirs(destination_folder, exist_ok=True)

# 파일 복사 및 이름 유지
for base_folder, destination_folder in zip(base_folders, destination_folders):
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    new_file_path = os.path.join(destination_folder, file_name)
                    shutil.copy2(file_path, new_file_path)

# 3. JSON 파일을 YOLO 형식으로 변환 후 제거
json_folder_path = '/data/ephemeral/home/dataset_yolo/labels/train'
output_folder = '/data/ephemeral/home/dataset_yolo/labels/train'

class_mapping = {
    "finger-1": 0, "finger-2": 1, "finger-3": 2, "finger-4": 3, "finger-5": 4,
    "finger-6": 5, "finger-7": 6, "finger-8": 7, "finger-9": 8, "finger-10": 9,
    "finger-11": 10, "finger-12": 11, "finger-13": 12, "finger-14": 13, "finger-15": 14,
    "finger-16": 15, "finger-17": 16, "finger-18": 17, "finger-19": 18,
    "Trapezoid": 19, "Trapezium": 20, "Capitate": 21, "Hamate": 22,
    "Pisiform": 23, "Triquetrum": 24, "Lunate": 25, "Scaphoid": 26,
    "Radius": 27, "Ulna": 28
}

for json_filename in os.listdir(json_folder_path):
    if json_filename.endswith('.json'):
        json_file_path = os.path.join(json_folder_path, json_filename)
        output_file_path = os.path.join(output_folder, f"{json_filename.split('.')[0]}.txt")

        with open(json_file_path, 'r') as f:
            data = json.load(f)

        with open(output_file_path, 'w') as f_out:
            for annotation in data["annotations"]:
                label = annotation["label"]
                if label in class_mapping:
                    class_index = class_mapping[label]
                else:
                    print(f"Label format error occurred: {label}. Skip.")
                    continue

                points = annotation["points"]
                points_str = ' '.join([f"{x} {y}" for x, y in points])
                f_out.write(f"{class_index} {points_str}\n")

        os.remove(json_file_path)  # JSON 파일 제거
        print(f"변환 및 제거 완료: {output_file_path}")

# 4. YOLO 형식으로 좌표 정규화
img_width, img_height = 2048, 2048
input_folder_path = '/data/ephemeral/home/dataset_yolo/labels/train'
output_folder_path = '/data/ephemeral/home/dataset_yolo/labels/train'

os.makedirs(output_folder_path, exist_ok=True)

for input_filename in os.listdir(input_folder_path):
    if input_filename.endswith('.txt'):
        input_file_path = os.path.join(input_folder_path, input_filename)
        output_file_path = os.path.join(output_folder_path, input_filename)

        with open(input_file_path, 'r') as file:
            lines = file.readlines()

        yolo_lines = []
        for line in lines:
            coords = list(map(float, line.split()))
            class_index = int(coords[0])
            x_coords = coords[1::2]
            y_coords = coords[2::2]

            x_coords_norm = [x / img_width for x in x_coords]
            y_coords_norm = [y / img_height for y in y_coords]

            points_str = ' '.join([f"{x} {y}" for x, y in zip(x_coords_norm, y_coords_norm)])
            yolo_line = f"{class_index} {points_str}\n"
            yolo_lines.append(yolo_line)

        with open(output_file_path, 'w') as file:
            file.writelines(yolo_lines)

        print(f"변환 완료: {output_file_path}")

print("모든 파일 변환 완료")

# 5. Train/Val 데이터셋 분할
def split_dataset(base_folder, train_folder, val_folder, split_ratio=0.8):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    image_files = natsorted(glob(os.path.join(base_folder, '*.*')))
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for file_path in train_files:
        shutil.move(file_path, os.path.join(train_folder, os.path.basename(file_path)))
    for file_path in val_files:
        shutil.move(file_path, os.path.join(val_folder, os.path.basename(file_path)))

    print(f"Train 이미지 개수: {len(train_files)}, Val 이미지 개수: {len(val_files)}")

# 이미지 분할
split_dataset('/data/ephemeral/home/dataset_yolo/images/train',
              '/data/ephemeral/home/dataset_yolo/images/train',
              '/data/ephemeral/home/dataset_yolo/images/val')

# 라벨 분할
split_dataset('/data/ephemeral/home/dataset_yolo/labels/train',
              '/data/ephemeral/home/dataset_yolo/labels/train',
              '/data/ephemeral/home/dataset_yolo/labels/val')

# 6. 사용 후 불필요한 폴더 삭제
shutil.rmtree('/data/ephemeral/home/dataset_yolo/train')
shutil.rmtree('/data/ephemeral/home/dataset_yolo/test/DCM')

