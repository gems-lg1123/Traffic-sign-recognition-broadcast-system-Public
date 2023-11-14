# Autonomous Vehicle Traffic Sign Detection and Recognition System Based on YOLO

This project focuses on the development of an autonomous vehicle traffic sign detection and recognition system using the YOLO (You Only Look Once) model. Our research and implementation cover the following key areas:

## 1. YOLO Model Study
- **Recognition Principle:** In-depth analysis of YOLO's object detection algorithm.
- **Modulation Methods:** Examination of the model's adjustment techniques for traffic sign recognition.
- **Operational Methods:** Overview of how the YOLO model is deployed in an autonomous driving context.

## 2. Image Preprocessing
- **Clarity, Brightness, and Color Bias Detection:** Metrics are checked against predefined thresholds.
- **Preprocessing Steps:**
  - **Correction:** Adjusting images that exceed threshold values.
  - **Histogram Equalization:** Applied to all images post-correction to enhance contrast.

### Preprocessing Impact:
- **Average Brightness:** Reduced from 0.094 to 0.090.
- **Average Color Bias:** Decreased from 0.714 to 0.120.
- **Average Clarity Adjustment:** Improved from 660.665 to 1071.064.

## 3. Audio Feedback Integration
- **Voice Pack Generation:** Based on the type of traffic sign detected.
- **Functionality:** Upon recognition of a sign, the corresponding audio file is played, providing vocal prompts.

## Performance Metrics
- **mAP (mean Average Precision):** Improved from 0.893 to 0.902 post-image preprocessing, indicating a near 1% increase in recognition accuracy.

### Observations:
- **Large and Clear Images:** Minor improvements in recognition.
- **Small and Less Clear Images:** Significant enhancement in detection and recognition accuracy.

## Conclusion
This research demonstrates the effectiveness of image preprocessing in improving the accuracy of traffic sign recognition using the YOLO model, particularly for smaller and less distinct signs.
