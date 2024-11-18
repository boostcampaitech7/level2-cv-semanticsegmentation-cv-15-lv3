import os
import json
import numpy as np
import SimpleITK as sitk
from skimage.draw import polygon
import matplotlib.pyplot as plt

# Paths
input_folder_path = '/data/ephemeral/home/dataset_nnunet/train/labels'  # JSON 파일이 있는 폴더
output_nifti_path = '/data/ephemeral/home/dataset_nnunet/train/train_labels_combined.nii.gz'  # 저장될 하나의 NIfTI 파일 경로
image_dimensions = (2048, 2048)  # 이미지 크기 (height, width)

# Step 1: Parse JSON and Extract Annotations
def parse_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    annotations = data.get("annotations", [])
    return annotations

# Step 2: Generate Labeled Mask from Annotations
def generate_mask_from_annotations(annotations, dimensions):
    """
    Create a labeled mask from polygon annotations.
    Each unique annotation gets a unique integer label.
    """
    mask = np.zeros(dimensions, dtype=np.uint16)  # uint16으로 더 많은 레이블 ID 허용
    for i, annotation in enumerate(annotations):
        points = np.array(annotation["points"], dtype=np.int32)  # 좌표를 NumPy 배열로 변환
        label = i + 1  # 고유 레이블 지정 (1부터 시작)
        rr, cc = polygon(points[:, 1], points[:, 0], dimensions)  # Row, column 좌표 생성
        mask[rr, cc] = label  # 마스크에 레이블 할당
    return mask

# Step 3: Save Mask as NIfTI File
def save_mask_as_nifti(volume, output_path):
    """
    Save a 3D volume as a NIfTI (.nii.gz) file.
    """
    sitk_image = sitk.GetImageFromArray(volume)  # NumPy 배열을 SimpleITK 이미지로 변환
    sitk_image.SetSpacing((1.0, 1.0, 1.0))  # 픽셀 간격 설정 (필요에 따라 조정)
    sitk.WriteImage(sitk_image, output_path)
    print(f"NIfTI file saved at: {output_path}")

# Step 4: Visualize Slices from the Combined NIfTI
def visualize_slices(volume, num_slices=5):
    """
    Visualize a few slices from the 3D volume.
    """
    slice_indices = np.linspace(0, volume.shape[0] - 1, num_slices, dtype=int)  # 균등한 간격으로 슬라이스 선택
    for idx in slice_indices:
        plt.imshow(volume[idx, :, :], cmap='nipy_spectral')  # 다양한 레이블 표시를 위한 컬러맵 사용
        plt.title(f"Slice {idx}")
        plt.axis("off")
        plt.show()

# Main Process
if __name__ == "__main__":
    # Get all JSON files in the folder
    json_files = sorted([f for f in os.listdir(input_folder_path) if f.endswith('.json')])

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in folder: {input_folder_path}")

    print(f"Found {len(json_files)} JSON files.")

    # Initialize a list to store 2D masks
    volume_slices = []

    # Process each JSON file
    for idx, json_file in enumerate(json_files):
        json_path = os.path.join(input_folder_path, json_file)

        # Parse the JSON file
        annotations = parse_json(json_path)

        # Generate a 2D labeled mask
        mask = generate_mask_from_annotations(annotations, image_dimensions)
        volume_slices.append(mask)

        print(f"Processed {json_file} into a 2D mask.")

    # Stack 2D masks into a 3D volume
    volume = np.stack(volume_slices, axis=0)  # (Depth, Height, Width)
    print(f"Combined volume shape: {volume.shape}")

    # Save the 3D volume as a single NIfTI file
    save_mask_as_nifti(volume, output_nifti_path)

    # Visualize a few slices from the combined volume
    visualize_slices(volume, num_slices=5)

    print("Conversion to single NIfTI file completed successfully!")
