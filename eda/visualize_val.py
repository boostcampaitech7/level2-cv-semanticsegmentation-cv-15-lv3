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
    이미지 이름에서 validation ID 추출
    """
    validation_ids = set()  # 중복 방지를 위해 set 사용
    
    # pred_df의 이미지 이름들 확인
    pred_image_names = pred_df['image_name'].unique()
    # st.sidebar.write("Number of unique images in pred_df:", len(pred_image_names))
    # st.sidebar.write("First few prediction image names:", pred_image_names[:5])
    
    # 실제 존재하는 ID 폴더들 확인
    dcm_folders = sorted(list(Path(image_base_path).glob('ID*')))
    # st.sidebar.write("Number of DCM folders found:", len(dcm_folders))
    # st.sidebar.write("First few DCM folders:", [f.name for f in dcm_folders[:5]])
    
    # 각 DCM 폴더에서 이미지 찾기
    for folder in dcm_folders:
        folder_id = folder.name[2:].zfill(3)  # 'ID166' -> '166'
        folder_images = list(folder.glob('*.png'))
        
        # 폴더 내 이미지들의 숫자 부분 추출
        for img_path in folder_images:
            img_number = ''.join(filter(str.isdigit, img_path.name))[:13]
            # pred_df의 모든 이미지 이름과 비교
            for pred_img_name in pred_image_names:
                if img_number in pred_img_name:  # 숫자 부분이 포함되어 있다면 매칭
                    validation_ids.add(folder_id)
                    break
            if folder_id in validation_ids:
                break
    
    result = sorted(list(validation_ids), key=lambda x: int(x))
    # st.sidebar.write("Found validation IDs:", result)
    return result

def main():
    st.set_page_config(layout="wide")
    st.title("Validation Results Visualization")

    # 세션 상태 초기화를 더 명확하게
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'selected_pred_csv' not in st.session_state:
        st.session_state.selected_pred_csv = None

    def increment_idx():
        st.session_state.current_idx = (st.session_state.current_idx + 1) % len(unique_images)
        
    def decrement_idx():
        st.session_state.current_idx = (st.session_state.current_idx - 1) % len(unique_images)

    # CSV 파일 로드
    script_dir = Path(__file__).parent.resolve()
    csv_dir = script_dir.parent / "eda" / "val_csv"
    image_base_path = script_dir.parent / "data" / "train" / "DCM"
    
    # 경로 디버깅
    # st.sidebar.write("CSV Directory:", csv_dir)
    # st.sidebar.write("Image Directory:", image_base_path)
    
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
        gt_df = pd.read_csv(gt_files[0])
        
        # validation ID 추출
        unique_images = get_validation_ids(pred_df, image_base_path)
        
        if not unique_images:
            st.error("No validation images found")
            return
            
        # 현재 인덱스가 범위를 벗어나지 않도록 조정
        st.session_state.current_idx = min(st.session_state.current_idx, len(unique_images) - 1)

        # 이전/다음 버튼과 선택 박스를 가로로 배치
        col1, col2, col3 = st.sidebar.columns([1,2,1])
        with col1:
            st.button("◀", on_click=decrement_idx, key="prev")
        
        with col2:
            selected_img = st.selectbox(
                "Select Image ID:",
                options=unique_images,
                index=st.session_state.current_idx,
                format_func=lambda x: f"ID{x}",
                key="image_select"
            )
        
        with col3:
            st.button("▶", on_click=increment_idx, key="next")
        
        if selected_img:
            st.session_state.current_idx = unique_images.index(selected_img)

        # Visualization Options
        st.sidebar.subheader("Visualization Options")
        show_labels = st.sidebar.checkbox("Show bone labels", value=True)
        show_gt = st.sidebar.checkbox("Show Ground Truth", value=True)
        show_pred = st.sidebar.checkbox("Show Prediction", value=True)
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
            id_folder = f"ID{selected_img}"
            folder_path = image_base_path / id_folder
            
            if folder_path.exists():
                img_files = sorted(list(folder_path.glob('*.png')))
                
                if len(img_files) == 2:
                    for i, img_file in enumerate(img_files):  # hand_type 제거
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                            # 이미지 매칭을 위한 숫자 추출
                            img_name = img_file.name
                            img_number = ''.join(filter(str.isdigit, img_name))[:13]
                            
                            # pred_df와 gt_df에서 해당 숫자를 포함하는 모든 이미지 이름 가져오기
                            matching_pred_images = sorted(list(set([  # 중복 제거
                                name for name in pred_df['image_name'].unique()
                                if ''.join(filter(str.isdigit, name))[:13] == img_number
                            ])))
                            matching_gt_images = sorted(list(set([  # 중복 제거
                                name for name in gt_df['image_name'].unique()
                                if ''.join(filter(str.isdigit, name))[:13] == img_number
                            ])))
                            
                            # GT와 예측을 같은 이미지에 표시
                            comparison_img = img.copy()
                            
                            # 매칭된 데이터 찾기
                            if matching_pred_images and matching_gt_images:
                                pred_image_name = matching_pred_images[0]
                                gt_image_name = matching_gt_images[0]
                                
                                pred_img_data = pred_df[pred_df['image_name'] == pred_image_name].copy()  # copy 추가
                                gt_img_data = gt_df[gt_df['image_name'] == gt_image_name].copy()  # copy 추가
                                
                                if not pred_img_data.empty and not gt_img_data.empty:
                                    # Ground Truth (파란색 계열)
                                    gt_color_map = {bone: [0, 0, 255] for bone in known_classes}
                                    if show_gt:
                                        comparison_img = visualize_bones_from_rle(
                                            comparison_img,
                                            gt_img_data,
                                            selected_bones,
                                            show_labels=False,
                                            thickness=line_thickness,
                                            color_map=gt_color_map
                                        )
                                    
                                    # Prediction (빨간색 계열)
                                    pred_color_map = {bone: [255, 0, 0] for bone in known_classes}
                                    if show_pred:
                                        comparison_img = visualize_bones_from_rle(
                                            comparison_img,
                                            pred_img_data,
                                            selected_bones,
                                            show_labels=show_labels,
                                            thickness=line_thickness,
                                            color_map=pred_color_map
                                        )
                                    
                                    # 범례 추가 (체크된 항목만 표시)
                                    legend_html = "<strong>Color Legend:</strong><br>"
                                    if show_gt:
                                        legend_html += "- <span style='color:blue'>Blue: Ground Truth</span><br>"
                                    if show_pred:
                                        legend_html += "- <span style='color:red'>Red: Prediction</span>"
                                    
                                    st.markdown(legend_html, unsafe_allow_html=True)
                                    
                                    # 이미지 표시
                                    st.image(comparison_img, use_container_width=True)
                                else:
                                    st.error(f"Empty data for image: {img_name}")
                            else:
                                st.error(f"Insufficient matching data for image: {img_name}")
                        else:
                            st.error(f"Failed to load image: {img_file}")

if __name__ == "__main__":
    main()