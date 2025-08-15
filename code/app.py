from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load models
tower_model = YOLO('tower_model.pt')
corrosion_model = YOLO('corrosion_model.pt')

# Paths to save results
SEGMENTED_IMAGES_PATH = 'static/segmented_images'
CORROSION_RESULTS_PATH = 'static/corrosion_results'
USER_ULOADS_PATH = 'static/uploads'
os.makedirs(SEGMENTED_IMAGES_PATH, exist_ok=True)
os.makedirs(CORROSION_RESULTS_PATH, exist_ok=True)
os.makedirs(USER_ULOADS_PATH, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect-corrosion', methods=['POST'])
def detect_corrosion():
    images = request.files.getlist('images')
    
    # Process uploaded images for tower segmentation
    for image in images:
        img_path = os.path.join(USER_ULOADS_PATH, image.filename)
        image.save(img_path)
        
        # Perform tower segmentation
        results = tower_model(img_path)
        original_image = cv2.imread(img_path)
        
        # Remove the background of the images using the results from the model
        for i, result in enumerate(results):
            if result.masks:
                for j, mask in enumerate(result.masks.data):
                    mask = mask.cpu().numpy().astype(np.uint8)
                    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
                    segmented_tower = cv2.bitwise_and(original_image, original_image, mask=mask_resized)
                    
                    # Save the segmented tower images
                    output_file = os.path.join(SEGMENTED_IMAGES_PATH, f"segmented_{i}_{j}_{image.filename}")
                    cv2.imwrite(output_file, segmented_tower)
    
    # Perform corrosion detection on segmented images and save them
    for img_file in os.listdir(SEGMENTED_IMAGES_PATH):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            segmented_path = os.path.join(SEGMENTED_IMAGES_PATH, img_file)
            results = corrosion_model(source=segmented_path, save=True, project=CORROSION_RESULTS_PATH)
            print(f"Processed {img_file}. Saved results in {CORROSION_RESULTS_PATH}.")

    # Cleanup both directories
    cleanup_directory(USER_ULOADS_PATH)
    cleanup_directory(SEGMENTED_IMAGES_PATH)
    
    return redirect(url_for('show_results'))

def cleanup_directory(directory):
    for img_file in os.listdir(directory):
        file_path = os.path.join(directory, img_file)
        if os.path.isfile(file_path):
            os.remove(file_path)

@app.route('/show-results', methods=['GET'])
def show_results():
    # Collect all image paths from subdirectories within the 'corrosion_results' folder
    predicted_images = []
    for root, _, files in os.walk(CORROSION_RESULTS_PATH):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                predicted_images.append(os.path.join(root, file))
    
    return render_template('results.html', images=predicted_images)

if __name__ == '__main__':
    app.run(debug=True)