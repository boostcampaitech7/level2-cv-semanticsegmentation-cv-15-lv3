import pandas as pd
import os

# 1. 엑셀 파일 읽기
file_path = "/home/hwang/leem/level2-cv-semanticsegmentation-cv-15-lv3/data/meta_data.xlsx"  # 수정할 엑셀 파일 경로
df = pd.read_excel(file_path)

# 2. "성별" 열에서 "_x008_" 제거
if "성별" in df.columns:
    df["성별"] = df["성별"].str.replace("_x0008_", "", regex=False).str.strip()
else:
    print("'성별' 열이 존재하지 않습니다.")

# 3. 수정된 데이터 저장
output_path = "meta_data.xlsx"  # 수정된 파일 저장 경로
df.to_excel(output_path, index=False)
print(f"수정된 파일이 저장되었습니다: {output_path}")

# ID 열 변환: 접두사 추가
df['ID'] = df['ID'].apply(lambda x: f"ID{x:03d}")  # 숫자를 3자리로 변환 후 접두사 추가

# 이미지 파일 경로 설정
base_paths = ["/home/hwang/leem/level2-cv-semanticsegmentation-cv-15-lv3/data/test/DCM", "/home/hwang/leem/level2-cv-semanticsegmentation-cv-15-lv3/data/train/DCM"]

for base_path in base_paths:
    # ID와 이미지 파일 매칭
    id_to_images = {}
    for id_dir in os.listdir(base_path):
        id_path = os.path.join(base_path, id_dir)
        if os.path.isdir(id_path):
            image_files = [f for f in os.listdir(id_path) if f.endswith(".png")]
            id_to_images[id_dir] = image_files

    # 데이터프레임 확장
    expanded_rows = []
    for _, row in df.iterrows():
        id_value = row['ID']
        images = id_to_images.get(id_value, [])
        for image_name in images:
            new_row = row.copy()
            new_row['image_name'] = image_name
            expanded_rows.append(new_row)

# 확장된 데이터프레임 생성
expanded_df = pd.DataFrame(expanded_rows)

# 결과 저장 (확장된 데이터를 새로운 파일로 저장)
output_path = "expanded_meta_data.xlsx"
expanded_df.to_excel(output_path, index=False)

# 사용자에게 알림
print(f"확장된 데이터가 {output_path}에 저장되었습니다.")