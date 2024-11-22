import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    return img.reshape(height, width)

def visualize_bones_from_rle(img, csv_data, selected_bones=None, show_labels=True, thickness=2, color_map=None):
    """RLE 데이터로부터 뼈를 시각화하는 함수"""
    vis_img = img.copy()
    height, width = img.shape[:2]
    
    # 색상 맵이 제공되지 않은 경우 생성
    if color_map is None:
        all_bone_names = sorted(list(set(csv_data['class'].unique())))
        color_map = {}
        colors = plt.cm.rainbow(np.linspace(0, 1, len(all_bone_names)))
        colors = (colors[:, :3] * 255).astype(np.uint8)
        
        for i, bone_name in enumerate(all_bone_names):
            color_map[bone_name] = colors[i].tolist()
    
    for bone in sorted(set(csv_data['class'].unique())):
        if selected_bones is None or bone in selected_bones:
            bone_data = csv_data[csv_data['class'] == bone]
            if not bone_data.empty:
                rle = bone_data.iloc[0]['rle']
                if pd.isna(rle):  # RLE이 없는 경우 건너뛰기
                    continue
                bone_mask = decode_rle_to_mask(rle, height, width)
                
                # 윤곽선 찾기
                contours, _ = cv2.findContours(bone_mask.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
                
                # 윤곽선 그리기
                if thickness == cv2.FILLED:
                    cv2.fillPoly(vis_img, contours, color_map[bone])
                else:
                    cv2.drawContours(vis_img, contours, -1, color_map[bone], thickness)
                
                if show_labels and contours:
                    # 가장 큰 윤곽선의 중심점 찾기
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        center = np.array([
                            int(M["m10"] / M["m00"]),
                            int(M["m01"] / M["m00"])
                        ])
                        
                        # 텍스트 배경 추가
                        text = bone
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness_text = 2
                        
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, font, font_scale, thickness_text
                        )
                        
                        # 텍스트 배경 박스 그리기
                        padding = 5
                        cv2.rectangle(vis_img, 
                                    (center[0] - text_width//2 - padding, 
                                     center[1] - text_height - padding),
                                    (center[0] + text_width//2 + padding, 
                                     center[1] + padding),
                                    (255, 255, 255),
                                    -1)
                        
                        # 텍스트 그리기
                        cv2.putText(vis_img, text,
                                  (center[0] - text_width//2, center[1]),
                                  font,
                                  font_scale,
                                  color_map[bone],
                                  thickness_text)
    
    return vis_img

def get_csv_files(csv_dir):
    """val_csv 디렉토리에서 CSV 파일들을 가져오는 함수"""
    csv_path = Path(csv_dir).resolve()  # 절대 경로로 변환
    
    # 디렉토리가 존재하는지 확인
    if not csv_path.exists():
        st.error(f"Directory not found: {csv_path}")
        return [], []
    
    # CSV 파일 찾기
    pred_files = sorted(list(csv_path.glob('*_val.csv')))
    gt_files = sorted(list(csv_path.glob('val_gt.csv')))
    
    return pred_files, gt_files

def get_image_path(image_id, base_path):
    """이미지 ID로 실제 이미지 경로 찾기"""
    # ID 폴더들을 순회
    for id_folder in Path(base_path).glob('ID*'):
        # 해당 ID 폴더 내의 모든 PNG 파일 확인
        for img_path in id_folder.glob('*.png'):
            if f"image_{image_id}" in img_path.name:
                return img_path
    return None

def get_validation_ids(pred_df, image_base_path):
    """
    DCM 폴더의 ID들을 탐색하여 validation 데이터가 있는 ID만 추출
    """
    validation_ids = []
    dcm_folders = sorted(list(Path(image_base_path).glob('ID*')))
    
    # 디버깅: 찾은 DCM 폴더들 출력
    st.sidebar.write("Found DCM folders:", [f.name for f in dcm_folders])
    
    # pred_df의 이미지 이름들을 set으로 변환
    pred_image_names = set(pred_df['image_name'].unique())
    
    # 디버깅: 예측 CSV의 이미지 이름들 출력
    st.sidebar.write("Prediction image names:", list(pred_image_names)[:5], "...")
    
    for folder in dcm_folders:
        # 각 ID 폴더의 이미지들 확인
        folder_images = list(folder.glob('*.png'))
        
        # 디버깅: 현재 폴더의 이미지들 출력
        st.sidebar.write(f"Images in {folder.name}:", [f.name for f in folder_images])
        
        for img_path in folder_images:
            # 디버깅: 이미지 이름 비교
            st.sidebar.write(f"Checking {img_path.name}")
            if img_path.name in pred_image_names:
                # ID 추출 (예: 'ID001' -> '001')
                id_num = folder.name[2:].zfill(3)  # 항상 3자리 숫자로 맞춤
                if id_num not in validation_ids:
                    validation_ids.append(id_num)
                    st.sidebar.write(f"Added ID: {id_num}")
                break
    
    result = sorted(validation_ids, key=lambda x: int(x))
    st.sidebar.write("Final validation IDs:", result)
    return result

def main():
    st.set_page_config(layout="wide")
    st.title("Validation Results Visualization")

    # 세션 상태 초기화
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'selected_pred_csv' not in st.session_state:
        st.session_state.selected_pred_csv = None

    # CSV 파일 로드
    script_dir = Path(__file__).parent.resolve()
    csv_dir = script_dir.parent / "eda" / "val_csv"
    image_base_path = script_dir.parent / "data" / "train" / "DCM"
    
    # 경로 디버깅
    st.sidebar.write("CSV Directory:", csv_dir)
    st.sidebar.write("Image Directory:", image_base_path)
    
    pred_files, gt_files = get_csv_files(csv_dir)
    
    if not pred_files:
        st.sidebar.error("No prediction CSV files found")
        return

    # Prediction CSV 선택
    st.sidebar.subheader("Select Prediction CSV")
    selected_pred = st.sidebar.selectbox(
        "Choose prediction file:",
        options=pred_files,
        format_func=lambda x: x.name if x else "",
        key='pred_select'
    )

    # CSV 내용 확인 및 디버깅
    if selected_pred and gt_files:
        pred_df = pd.read_csv(selected_pred)
        st.sidebar.write("CSV columns:", pred_df.columns)
        st.sidebar.write("Sample data:", pred_df[['image_name', 'class']].head())

        # validation ID 추출
        unique_images = get_validation_ids(pred_df, image_base_path)
        
        if not unique_images:
            st.error("No validation images found")
            return
            
        # 현재 인덱스가 범위를 벗어나지 않도록 조정
        st.session_state.current_idx = min(st.session_state.current_idx, len(unique_images) - 1)

        # 이미지 선택 (사이드바로 이동)
        st.sidebar.subheader("Image Selection")
        selected_img = st.sidebar.selectbox(
            "Select Image ID:",
            options=unique_images,
            index=st.session_state.current_idx,
            format_func=lambda x: f"ID{x}"
        )
        
        if selected_img:
            st.session_state.current_idx = unique_images.index(selected_img)

        # Visualization Options
        st.sidebar.subheader("Visualization Options")
        show_labels = st.sidebar.checkbox("Show bone labels", value=True)
        show_contours = st.sidebar.checkbox("Show contours", value=True)
        show_mask = st.sidebar.checkbox("Show mask", value=False)
        
        if show_mask:
            mask_opacity = st.sidebar.slider("Mask opacity", 0.0, 1.0, 0.5)
        else:
            mask_opacity = 0.5
            
        line_thickness = st.sidebar.slider("Line thickness", 1, 5, 2)
        
        # 뼈 선택
        st.sidebar.subheader("Bone Selection")
        known_classes = sorted(list(set(pred_df['class'].unique())))
        selected_bones = st.sidebar.multiselect(
            "Select bones to visualize:",
            known_classes,
            default=known_classes
        )

        if selected_img:
            # ID 폴더에서 이미지 찾기
            id_folder = f"ID{selected_img}"  # 이미 올바른 형식
            folder_path = image_base_path / id_folder
            
            if folder_path.exists():
                # 해당 ID 폴더의 이미지들 로드
                img_files = sorted(list(folder_path.glob('*.png')))
                
                if len(img_files) == 2:
                    # 결과 시각화를 위한 컬럼 생성
                    col1, col2 = st.columns(2)
                    
                    for i, (hand_type, img_file) in enumerate(zip(['Right', 'Left'], img_files)):
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                            # 예측과 GT 데이터 필터링 (파일 이름으로 정확히 매칭)
                            img_name = img_file.name
                            pred_img_data = pred_df[pred_df['image_name'] == img_name]
                            gt_img_data = gt_df[gt_df['image_name'] == img_name]
                            
                            # 결과 시각화
                            col = col1 if i == 0 else col2
                            with col:
                                st.subheader(f"{hand_type} Hand")
                                
                                # 원본 이미지
                                st.markdown("**Original Image**")
                                st.image(img, use_container_width=True)
                                
                                # Prediction
                                st.markdown("**Prediction**")
                                pred_img = img.copy()
                                if show_mask:
                                    # 마스크 생성 (검은 배경에 마스크만)
                                    mask_img = visualize_bones_from_rle(
                                        np.zeros_like(img),
                                        pred_img_data,
                                        selected_bones,
                                        show_labels=False,
                                        thickness=cv2.FILLED
                                    )
                                    # 마스크와 원본 이미지 블렌딩
                                    pred_img = cv2.addWeighted(pred_img, 1, mask_img, mask_opacity, 0)
                                
                                if show_contours:
                                    # 윤곽선 추가
                                    pred_img = visualize_bones_from_rle(
                                        pred_img,
                                        pred_img_data,
                                        selected_bones,
                                        show_labels=show_labels,
                                        thickness=line_thickness
                                    )
                                
                                # 수정된 이미지가 있을 때만 표시
                                if show_mask or show_contours:
                                    st.image(pred_img, use_container_width=True)
                                
                                # Ground Truth
                                st.markdown("**Ground Truth**")
                                gt_img = img.copy()
                                if show_mask:
                                    mask_img = visualize_bones_from_rle(
                                        np.zeros_like(img),
                                        gt_img_data,
                                        selected_bones,
                                        show_labels=False,
                                        thickness=cv2.FILLED
                                    )
                                    gt_img = cv2.addWeighted(gt_img, 1, mask_img, mask_opacity, 0)
                                
                                if show_contours:
                                    gt_img = visualize_bones_from_rle(
                                        gt_img,
                                        gt_img_data,
                                        selected_bones,
                                        show_labels=show_labels,
                                        thickness=line_thickness
                                    )
                                
                                # 수정된 이미지가 있을 때만 표시
                                if show_mask or show_contours:
                                    st.image(gt_img, use_container_width=True)
                        else:
                            st.error(f"Failed to load image: {img_file}")
                else:
                    st.error(f"Expected 2 images in folder {id_folder}, found {len(img_files)}")
            else:
                st.error(f"Folder not found: {folder_path}")

if __name__ == "__main__":
    main()