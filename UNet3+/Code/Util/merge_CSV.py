import pandas as pd

# 1번 CSV와 2번 CSV 파일 경로
csv1_path = "/data/ephemeral/home/MCG/UNetRefactored/CSV/0971.csv"
csv2_path = "/data/ephemeral/home/MCG/UNetRefactored/CSV/connect!!!!!!!!!!!!.csv"

# CSV 읽기
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# 중복 판단 키 생성
df1["key"] = df1["image_name"] + "_" + df1["class"]
df2["key"] = df2["image_name"] + "_" + df2["class"]

# 2번 CSV를 딕셔너리로 변환
df2_dict = df2.set_index("key").to_dict(orient="index")

# 1번 CSV 순서를 유지하며 데이터 대체
updated_rows = []
for _, row in df1.iterrows():
    key = row["key"]
    if key in df2_dict:
        # df2_dict에서 대체 데이터 추출
        updated_row = {
            "image_name": row["image_name"],
            "class": row["class"],
            "rle": df2_dict[key]["rle"]
        }
        updated_rows.append(updated_row)
    else:
        updated_rows.append(row.to_dict())  # 기존 데이터를 dict 형식으로 추가

# DataFrame 생성
updated_df = pd.DataFrame(updated_rows)

# key 컬럼 삭제
updated_df.drop(columns=["key"], inplace=True)

# 결과 CSV 저장
output_path = "/data/ephemeral/home/MCG/UNetRefactored/CSV/NonNonConnection.csv"
updated_df.to_csv(output_path, index=False)

print(f"결과 CSV가 '{output_path}'에 저장되었습니다.")
