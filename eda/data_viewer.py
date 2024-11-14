import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
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
            uploaded_file = st.file_uploader("Upload test CSV file", type=['csv'])
            if uploaded_file is not None:
                st.session_state.test_csv = pd.read_csv(uploaded_file)
                st.success("Test CSV file loaded successfully!")
        
        # 기본 경로 설정
        base_path = f"../data/{dataset_type.lower()}/outputs_json"
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
            
            # 뼈 선택 (맨 아래로 이동)
            if selected_id:
                st.subheader("Bone Selection")
                hand_data = read_hand_jsons(selected_id)
                
                # 모든 가능한 뼈 목록 생성
                all_bones = set()
                for hand_info in hand_data.values():
                    if hand_info is not None and 'annotations' in hand_info:
                        all_bones.update(hand_info['annotations'].keys())
                
                if all_bones:
                    selected_bones = st.multiselect(
                        "Select bones to visualize:",
                        sorted(list(all_bones)),
                        default=sorted(list(all_bones))
                    )
            
    # 메인 영역에 이미지 표시
    if selected_id:
        col1, col2 = st.columns(2)

        if dataset_type == "Train" or (dataset_type == "Test" and st.session_state.test_csv is not None):
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
            # Test 데이터셋이고 CSV가 없는 경우 이미지만 표시
            for hand_type in ['Left', 'Right']:
                img_path = str(Path(base_path).parent.parent / 'test/DCM' / selected_id.name / f"{selected_id.name}_{hand_type[0]}.png")
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if hand_type == 'Left':
                        with col1:
                            st.subheader(f"{hand_type} Hand")
                            st.image(img, use_column_width=True)
                    else:
                        with col2:
                            st.subheader(f"{hand_type} Hand")
                            st.image(img, use_column_width=True)

if __name__ == "__main__":
    main()