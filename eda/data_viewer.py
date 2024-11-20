import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
import os
from pathlib import Path
import glob

def get_id_folders(base_path):
    """outputs_json 디렉토리 내의 모든 ID 폴더를 가져오는 함수"""
    return sorted([p for p in Path(base_path).glob('ID*')])

def read_hand_jsons(id_folder):
    """ID 폴더 내의 왼손/오른손 JSON 파일을 읽는 함수"""
    jsons = {'Left': None, 'Right': None}  # 기본값 설정
    
    json_files = sorted(list(id_folder.glob('*.json')))
    
    if len(json_files) == 2:  # 파일이 2개인 경우
        if '_L.' in json_files[0].name or '_R.' in json_files[0].name:
            # ID001 형식: _L, _R로 구분
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                hand_type = 'Left' if '_L.' in str(json_file) else 'Right'
                jsons[hand_type] = {
                    'image': {
                        'width': data['metadata']['width'],
                        'height': data['metadata']['height'],
                        'filename': data['filename']
                    },
                    'annotations': {
                        ann['label']: {
                            'points': ann['points'],
                            'type': ann['type']
                        }
                        for ann in data['annotations']
                    }
                }
        else:
            # 나머지 ID: 첫 번째 파일이 Right, 두 번째 파일이 Left
            for i, json_file in enumerate(json_files):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                hand_type = 'Right' if i == 0 else 'Left'
                jsons[hand_type] = {
                    'image': {
                        'width': data['metadata']['width'],
                        'height': data['metadata']['height'],
                        'filename': data['filename']
                    },
                    'annotations': {
                        ann['label']: {
                            'points': ann['points'],
                            'type': ann['type']
                        }
                        for ann in data['annotations']
                    }
                }
    
    return jsons

def visualize_bones(img, annotations, selected_bones=None, show_labels=True, thickness=2):
    """뼈 윤곽선을 시각화하는 함수"""
    vis_img = img.copy()
    
    if not annotations:  # annotations이 None인 경우 처리
        return vis_img
    
    # 모든 가능한 뼈 이름으로 고정된 색상 맵 생성
    all_bone_names = sorted(list(annotations.keys()))
    color_map = {}
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_bone_names)))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    
    for i, bone_name in enumerate(all_bone_names):
        color_map[bone_name] = colors[i].tolist()
    
    for bone_name, bone_data in annotations.items():
        if selected_bones is None or bone_name in selected_bones:
            points = np.array(bone_data['points'], dtype=np.int32)
            if points.size > 0:  # points가 비어있지 않은 경우에만 처리
                # 윤곽선 그리기
                if thickness == cv2.FILLED:
                    cv2.fillPoly(vis_img, [points], color_map[bone_name])
                else:
                    cv2.polylines(vis_img, [points], 
                                isClosed=True, 
                                color=color_map[bone_name], 
                                thickness=thickness)
                
                if show_labels:
                    # 텍스트 배경 추가
                    center = points.mean(axis=0).astype(int)
                    text = bone_name
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness_text = 2
                    
                    # 텍스트 크기 계산
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
                                (255, 255, 255),  # 흰색 배경
                                -1)  # 채우기
                    
                    # 텍스트 그리기
                    cv2.putText(vis_img, text,
                               (center[0] - text_width//2, center[1]),
                               font,
                               font_scale,
                               color_map[bone_name],  # 뼈와 같은 색상 사용
                               thickness_text)
    
    return vis_img

def visualize_bones_from_rle(img, csv_data, selected_bones=None, show_labels=True, thickness=2):
    """RLE 데이터로부터 뼈를 시각화하는 함수"""
    vis_img = img.copy()
    height, width = img.shape[:2]
    
    # 모든 가능한 뼈 이름으로 고정된 색상 맵 생성
    all_bone_names = sorted(list(set(csv_data['class'].unique())))
    color_map = {}
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_bone_names)))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    
    for i, bone_name in enumerate(all_bone_names):
        color_map[bone_name] = colors[i].tolist()
    
    for bone in all_bone_names:
        if selected_bones is None or bone in selected_bones:
            bone_data = csv_data[csv_data['class'] == bone]
            if not bone_data.empty:
                rle = bone_data.iloc[0]['rle']
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
                
                # 라벨 표시
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
                        
                        # 텍스트 크기 계산
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
                                    (255, 255, 255),  # 흰색 배경
                                    -1)  # 채우기
                        
                        # 텍스트 그리기
                        cv2.putText(vis_img, text,
                                   (center[0] - text_width//2, center[1]),
                                   font,
                                   font_scale,
                                   color_map[bone],  # 뼈와 같은 색상 사용
                                   thickness_text)
    
    return vis_img

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    return img.reshape(height, width)

def main():
    st.set_page_config(layout="wide")
    st.title("Hand Bone X-ray Visualization")
    
    # 세션 상태 초기화
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'show_mask' not in st.session_state:
        st.session_state.show_mask = False
    if 'test_csv' not in st.session_state:
        st.session_state.test_csv = None
    if 'dataset_type' not in st.session_state:
        st.session_state.dataset_type = "Train"
    
    with st.sidebar:
        st.header("Controls")
        
        # Dataset 선택
        dataset_type = st.radio(
            "Select Dataset:",
            ["Train", "Test"],
            horizontal=True,
            key='dataset_type'
        )
        
        # 데이터셋이 변경되면 인덱스 초기화
        if dataset_type != st.session_state.dataset_type:
            st.session_state.current_idx = 0
            st.session_state.dataset_type = dataset_type
        
        # Test 데이터셋인 경우 CSV 파일 업로드
        if dataset_type == "Test":
            csv_folder = "./csv"  # CSV 파일들이 있는 폴더 경로
            csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
            
            selected_csv = st.selectbox(
                "Select CSV file:",
                options=csv_files,
                index=0 if csv_files else None
            )
            
            if selected_csv:
                csv_path = os.path.join(csv_folder, selected_csv)
                st.session_state.test_csv = pd.read_csv(csv_path)
                st.success(f"Loaded: {selected_csv}")
        
        # base_path 수정 (test의 경우 json 폴더 대신 다른 경로 사용)
        if dataset_type == "Train":
            base_path = "../data/train/outputs_json"
        else:
            base_path = "../data/test/DCM"  # test 이미지가 있는 경로

        id_folders = get_id_folders(base_path)
        
        if not id_folders:  # 폴더가 비어있는 경우
            st.error(f"No ID folders found in {base_path}")
            return
        
        # 현재 인덱스가 유효한지 확인
        st.session_state.current_idx = min(st.session_state.current_idx, len(id_folders) - 1)
        
        # ID 선택 영역
        st.subheader("ID Selection")
        col1, col2, col3 = st.columns([1,3,1])
        
        with col1:
            if st.button('⬅️') and len(id_folders) > 0:
                st.session_state.current_idx = (st.session_state.current_idx - 1) % len(id_folders)
                
        with col2:
            selected_id = st.selectbox(
                "Select ID:",
                options=id_folders,
                index=st.session_state.current_idx,
                format_func=lambda x: x.name if x else "",
                key='id_select'
            )
            if selected_id:  # selected_id가 None이 아닌 경우에만 인덱스 업데이트
                st.session_state.current_idx = id_folders.index(selected_id)
            
        with col3:
            if st.button('➡️') and len(id_folders) > 0:
                st.session_state.current_idx = (st.session_state.current_idx + 1) % len(id_folders)
        
        # Visualization Options (train 또는 test+csv인 경우에만 표시)
        if dataset_type == "Train" or (dataset_type == "Test" and st.session_state.test_csv is not None):
            st.subheader("Visualization Options")
            show_labels = st.checkbox("Show bone labels", value=True)
            show_contours = st.checkbox("Show contours", value=True)
            show_mask = st.checkbox("Show mask", value=False)
            
            if show_mask:
                mask_opacity = st.slider("Mask opacity", 0.0, 1.0, 0.5)
            else:
                mask_opacity = 0.5
                
            line_thickness = st.slider("Line thickness", 1, 5, 2)
            
            # 뼈 선택
            st.subheader("Bone Selection")
            known_classes = ['Capitate', 'Hamate', 'Lunate', 'Pisiform', 'Radius',
                           'Scaphoid', 'Trapezium', 'Trapezoid', 'Triquetrum', 'Ulna'] + \
                          [f'finger-{i}' for i in range(1, 20)]
            selected_bones = st.multiselect(
                "Select bones to visualize:",
                sorted(known_classes),
                default=sorted(known_classes)
            )
        else:
            show_mask = False
            show_labels = False
            show_contours = False
            mask_opacity = 0.5
            line_thickness = 2
            selected_bones = []
            
    # 메인 영역에 이미지 표시
    if selected_id:
        col1, col2 = st.columns(2)

        if dataset_type == "Train":
            hand_data = read_hand_jsons(selected_id)
        
            for hand_type, data in hand_data.items():
                if data is None:  # 데이터가 없는 경우 건너뛰기
                    continue
                    
                # 이미지 파일명 처리
                img_filename = data['image']['filename'].split('.')[0] + '.png'
                img_path = str(Path(base_path).parent.parent / 'train/DCM' / selected_id.name / img_filename)
                
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 마스크 생성
                    if show_mask and 'annotations' in data:
                        mask = np.zeros_like(img)
                        mask_img = visualize_bones(
                            mask,
                            data['annotations'],
                            selected_bones,
                            show_labels=False,
                            thickness=cv2.FILLED
                        )
                        # 마스크 블렌딩
                        img = cv2.addWeighted(img, 1, mask_img, mask_opacity, 0)
                    
                    # 윤곽선과 라벨 표시
                    if show_contours and 'annotations' in data:
                        img = visualize_bones(
                            img,
                            data['annotations'],
                            selected_bones,
                            show_labels=show_labels,
                            thickness=line_thickness
                        )
                    
                    if hand_type == 'Left':
                        with col1:
                            st.subheader(f"{hand_type} Hand")
                            st.image(img, use_column_width=True)
                    else:
                        with col2:
                            st.subheader(f"{hand_type} Hand")
                            st.image(img, use_column_width=True)
                else:
                    st.error(f"Cannot load image: {img_path}")

        else:
            for hand_type in ['Right', 'Left']:
                img_files = list(Path(base_path).glob(f"{selected_id.name}/*.png"))
                
                if len(img_files) > 0:
                    img_path = str(img_files[0] if hand_type == 'Right' else img_files[1])
                    img_filename = Path(img_path).name
                    
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # CSV가 로드되었을 때만 마스크와 윤곽선 표시
                        if st.session_state.test_csv is not None:
                            height, width = img.shape[:2]
                            csv_data = st.session_state.test_csv[
                                st.session_state.test_csv['image_name'] == img_filename
                            ]
                            
                            result_img = img.copy()
                            
                            # 마스크 처리
                            if show_mask:
                                mask_img = visualize_bones_from_rle(
                                    np.zeros_like(img),
                                    csv_data,
                                    selected_bones,
                                    show_labels=False,
                                    thickness=cv2.FILLED
                                )
                                result_img = cv2.addWeighted(result_img, 1, mask_img, mask_opacity, 0)
                            
                            # 윤곽선과 라벨 처리
                            if show_contours:
                                result_img = visualize_bones_from_rle(
                                    result_img,
                                    csv_data,
                                    selected_bones,
                                    show_labels=show_labels,
                                    thickness=line_thickness
                                )
                            
                            img = result_img
                        
                        # 이미지 표시
                        if hand_type == 'Left':
                            with col1:
                                st.subheader(f"{hand_type} Hand")
                                st.image(img, use_column_width=True)
                        else:
                            with col2:
                                st.subheader(f"{hand_type} Hand")
                                st.image(img, use_column_width=True)
                    else:
                        st.error(f"Cannot load image: {img_path}")

if __name__ == "__main__":
    main()