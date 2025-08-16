# Structural Defects Detection – Tower Corrosion

This project detects **structural defects** in metallic towers, particularly **corrosion**, using deep learning object detection and segmentation models YOLOv11.  

## Project Overview
- **Goal**: Automate the inspection of metallic structures to improve efficiency, reduce human error, and enhance safety.
- **Defects Detected**: Corrosion.
- **Key Tasks**:
  - Tower segmentation
  - Corrosion detection
  - Prototype web-based demonstrator for engineers to upload images and view results


## Model Performance

### YOLOv11M — Tower Segmentation
**Box metrics**

| Metric            | Value  |
|-------------------|:------:|
| mAP@0.5 (Box)     | 0.9950 |
| mAP@0.5:0.95 (Box)| 0.9924 |
| Precision (Box)   | 0.9987 |
| Recall (Box)      | 0.9913 |

**Mask metrics**

| Metric             | Value  |
|--------------------|:------:|
| mAP@0.5 (Mask)     | 0.9950 |
| mAP@0.5:0.95 (Mask)| 0.9327 |
| Precision (Mask)   | 0.9987 |
| Recall (Mask)      | 0.9913 |

**Overall Fitness:** 1.9316

---

### YOLOv11M — Corrosion Object Detection
| Metric        | Value  |
|---------------|:------:|
| Precision     | 0.4008 |
| Recall        | 0.4286 |
| F1 Score      | 0.4142 |
| mAP@0.5       | 0.3259 |
| mAP@0.5:0.95  | 0.1478 |

## Dataset
- **Source**: 573 high-resolution drone images of a metallic tower with corrosion defects
- **Split**:
  - Tower Dataset: 70% train / 20% val / 10% test
  - Segmented Tower Dataset: 87% train / 8% val / 4% test

## Installation & Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/solar11781/tower-corrosion-detection.git
   cd tower-corrosion-detection/code
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask web app:
   ```bash
   python app.py
   ```

4. Access the app:
   - Open `http://127.0.0.1:5000` in your browser.
   - Upload images for defect detection.

## Demo
[Project Demo Video](https://youtu.be/lAMwCdQ0eAQ)
