import json
import os

# JSON 파일들이 있는 폴더 경로
json_folder_path = '/data/ephemeral/home/dataset_yolo/train/labels_ori'
# YOLO11 형식으로 저장할 폴더 경로
output_folder = '/data/ephemeral/home/dataset_yolo/train/labels'

# 클래스 매핑 딕셔너리 (한 칸씩 밀림)
class_mapping = {
    "finger-1": 0, "finger-2": 1, "finger-3": 2, "finger-4": 3, "finger-5": 4,
    "finger-6": 5, "finger-7": 6, "finger-8": 7, "finger-9": 8, "finger-10": 9,
    "finger-11": 10, "finger-12": 11, "finger-13": 12, "finger-14": 13, "finger-15": 14,
    "finger-16": 15, "finger-17": 16, "finger-18": 17, "finger-19": 18,
    "Trapezoid": 19, "Trapezium": 20, "Capitate": 21, "Hamate": 22,
    "Pisiform": 23, "Triquetrum": 24, "Lunate": 25, "Scaphoid": 26,
    "Radius": 27, "Ulna": 28
}

# JSON 폴더 내 모든 JSON 파일 처리
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

        print(f"변환 완료: {output_file_path}")
