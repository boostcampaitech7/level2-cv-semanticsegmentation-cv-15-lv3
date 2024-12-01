import os

# 이미지 크기 설정
img_width, img_height = 2048, 2048

# 입력과 출력 폴더 경로 설정
input_folder_path = '/data/ephemeral/home/dataset_yolo/labels/val_1'  # 입력 폴더 경로
output_folder_path = '/data/ephemeral/home/dataset_yolo/labels/val'  # 출력 폴더 경로

# 출력 폴더가 없으면 생성
os.makedirs(output_folder_path, exist_ok=True)

# 입력 폴더의 모든 .txt 파일 처리
for input_filename in os.listdir(input_folder_path):
    if input_filename.endswith('.txt'):
        input_file_path = os.path.join(input_folder_path, input_filename)
        output_file_path = os.path.join(output_folder_path, input_filename)  # 출력 파일 경로 설정
        
        # 파일 읽기 및 변환
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
        
        yolo_lines = []
        for line in lines:
            # 각 줄에서 좌표 추출 (첫 번째 값은 클래스 인덱스라고 가정)
            coords = list(map(float, line.split()))
            class_index = int(coords[0])
            x_coords = coords[1::2]
            y_coords = coords[2::2]
            
            # 각 좌표를 이미지 크기로 정규화
            x_coords_norm = [x / img_width for x in x_coords]
            y_coords_norm = [y / img_height for y in y_coords]
            
            # YOLO 형식으로 저장할 내용: <class-index> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ... <xn_norm> <yn_norm>
            points_str = ' '.join([f"{x} {y}" for x, y in zip(x_coords_norm, y_coords_norm)])
            yolo_line = f"{class_index} {points_str}\n"
            yolo_lines.append(yolo_line)
        
        # 변환된 내용을 새 파일에 저장
        with open(output_file_path, 'w') as file:
            file.writelines(yolo_lines)
        
        print(f"변환 완료: {output_file_path}")

print("모든 파일 변환 완료")
