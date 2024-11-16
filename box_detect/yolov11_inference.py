import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("/data/ephemeral/home/MCG/runs/detect/train/weights/best.pt")

# Define the root directory containing the DCM folder
root_dir = "/data/ephemeral/home/MCG/data/test/DCM"

# Iterate through each patient folder in the DCM directory
for patient_id in os.listdir(root_dir):
    patient_dir = os.path.join(root_dir, patient_id)
    if os.path.isdir(patient_dir):
        # Process each image in the patient's folder
        for image_file in os.listdir(patient_dir):
            if image_file.endswith(".png"):  # Check if it's a PNG image
                image_path = os.path.join(patient_dir, image_file)
                print(f"Running inference on: {image_path}")
                # Run inference and save results
                model.predict(image_path, save=True, imgsz=2048, conf=0.5,max_det=3,augment=True)
