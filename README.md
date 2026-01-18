# 😷 Face Mask Detection System with Liveness Verification

An AI-powered Face Mask Detection web application with smart liveness verification
to prevent spoofing using photos, dummy faces, or static images.

The system uses Deep Learning (CNN) for mask detection and Computer Vision
techniques like eye blink and head movement detection for liveness verification.

---

## Features

- Face Mask Detection using CNN
- Eye Blink based Liveness Detection
- Head Movement Detection (Left / Right)
- Covered Eyes / Face Warning
- Image Upload Detection
- Real-time Webcam Detection
- Prevents photo and dummy face spoofing
- Interactive Web App using Streamlit

---

##  Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- Git & Git LFS

---

##  Project Structure
face-mask-detection-streamlit/
│
├── app.py # Main Streamlit application
├── mask_detector_model.h5 # Trained CNN model (managed using Git LFS)
├── haarcascade_eye.xml # Eye detection Haar Cascade
├── requirements.txt # Project dependencies
└── README.md # Project documentation

