import os
import shutil

# Source folder path and destination folder path to copy
original_folder = '/data/ephemeral/home/dataset'
copied_folder = '/data/ephemeral/home/dataset_nnunet'

# Copy the entire source folder to the destination folder
if not os.path.exists(copied_folder):  # Copy only if the destination folder does not exist
    shutil.copytree(original_folder, copied_folder)
    print(f"{original_folder} was successfully copied to {copied_folder}.")

# Original parent folder paths containing files to copy
base_folders = [
    '/data/ephemeral/home/dataset/train/DCM',
    '/data/ephemeral/home/dataset/train/outputs_json',
    '/data/ephemeral/home/dataset/test/DCM'
]

# Destination parent folder paths to copy files to
destination_folders = [
    '/data/ephemeral/home/dataset_nnunet/train/images',
    '/data/ephemeral/home/dataset_nnunet/train/labels',
    '/data/ephemeral/home/dataset_nnunet/test'
]

# Create destination folder (ignore if it already exists)
for destination_folder in destination_folders:
    os.makedirs(destination_folder, exist_ok=True)

# Repeat for each pair of source and destination folders
for base_folder, destination_folder in zip(base_folders, destination_folders):
    # Search all subfolders within the original parent folder
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        # Check if it is a folder
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                # Check if it is a file
                if os.path.isfile(file_path):
                    # Maintain the original file name
                    new_file_path = os.path.join(destination_folder, file_name)
                    # Copy the file
                    shutil.copy2(file_path, new_file_path)


shutil.rmtree('/data/ephemeral/home/dataset_nnunet/test/DCM')
shutil.rmtree('/data/ephemeral/home/dataset_nnunet/train/DCM')
shutil.rmtree('/data/ephemeral/home/dataset_nnunet/train/outputs_json')
