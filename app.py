# ================= IMPORTS =================
import os
import subprocess
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

# ================= DEVELOPER NAME =================
DEVELOPER_NAME = "Ronak Gupta"

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="😷",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.title {font-size:38px;font-weight:700;color:#0f172a;}
.subtitle {font-size:18px;color:#475569;margin-bottom:8px;}
.dev {font-size:14px;color:#64748b;margin-bottom:20px;}
.instruction {background:#f1f5f9;padding:12px;border-radius:8px;font-size:16px;margin-bottom:15px;}
.footer {text-align:center;color:#64748b;margin-top:30px;}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
MODEL_PATH = "mask_detector_model.h5"

# Ensure Git LFS files are pulled on Streamlit Cloud
if not os.path.exists(MODEL_PATH):
    try:
        subprocess.run(["git", "lfs", "pull"], check=True)
    except Exception as e:
        st.error("❌ Model file not found. Git LFS pull failed.")
        st.stop()

model = load_model(MODEL_PATH)


# ================= LOAD CASCADES =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# ================= HEADER =================
st.markdown(
    f"""
    <div class="title">😷 Face Mask Detection System</div>
    <div class="subtitle">Mask Detection with Smart Liveness Verification</div>
    <div class="dev">Developed by <b>{DEVELOPER_NAME}</b></div>
    """,
    unsafe_allow_html=True
)

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Control Panel")
mode = st.sidebar.radio(
    "Select Mode",
    ["📷 Image Upload", "🎥 Live Webcam (Liveness)", "ℹ️ About"]
)

# =====================================================
# 📷 IMAGE UPLOAD MODE
# =====================================================
if mode == "📷 Image Upload":

    uploaded_file = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100)
        )

        if len(faces) == 0:
            st.warning("⚠️ No clear face detected. Please upload a frontal face image.")
        else:
            # ✅ ONLY LARGEST FACE
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            face = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (128, 128)) / 255.0
            face_resized = face_resized.reshape(1, 128, 128, 3)

            pred = model.predict(face_resized)[0]
            mask_prob = pred[1]
            no_mask_prob = pred[0]

            if mask_prob > no_mask_prob:
                label = f"Mask 😷 ({mask_prob*100:.1f}%)"
                color = (0, 255, 0)
            else:
                label = f"No Mask ❌ ({no_mask_prob*100:.1f}%)"
                color = (0, 0, 255)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            st.image(img, use_container_width=True)

# =====================================================
# 🎥 LIVE WEBCAM WITH SMART LIVENESS
# =====================================================
elif mode == "🎥 Live Webcam (Liveness)":

    st.markdown(
        '<div class="instruction">👉 Please look at the camera, blink your eyes and move your head slightly left or right.</div>',
        unsafe_allow_html=True
    )

    start = st.checkbox("▶️ Start Webcam")

    blink_detected = False
    movement_detected = False
    last_eye_count = 2
    last_face_x = None
    last_action_time = time.time()

    if start:
        cap = cv2.VideoCapture(0)
        frame_box = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100)
            )

            status = "NO LIVENESS ❌"
            status_color = (0, 0, 255)
            instruction = "Blink eyes & move head"

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

                # 👁️ Blink detection
                if len(eyes) < last_eye_count:
                    blink_detected = True
                    last_action_time = time.time()
                last_eye_count = len(eyes)

                # 👤 Head movement detection
                if last_face_x is not None and abs(x - last_face_x) > 15:
                    movement_detected = True
                    last_action_time = time.time()
                last_face_x = x

                # 🚨 Covered eyes warning
                if len(eyes) == 0:
                    instruction = "Eyes not visible! Please uncover eyes."

                # ✅ Liveness verified
                if (blink_detected or movement_detected) and time.time() - last_action_time < 4:
                    status = "REAL HUMAN ✅"
                    status_color = (0, 255, 0)
                    instruction = "Liveness verified"

                    face = frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (128, 128)) / 255.0
                    face_resized = face_resized.reshape(1, 128, 128, 3)

                    pred = model.predict(face_resized)[0]
                    mask_prob = pred[1]
                    no_mask_prob = pred[0]

                    if mask_prob > no_mask_prob:
                        label = f"Mask 😷 ({mask_prob*100:.1f}%)"
                        color = (0, 255, 0)
                    else:
                        label = f"No Mask ❌ ({no_mask_prob*100:.1f}%)"
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.putText(frame, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(frame, instruction, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            frame_box.image(frame, channels="BGR")

        cap.release()

# =====================================================
# ℹ️ ABOUT
# =====================================================
else:
    st.write(f"""
    ### About This Project

    ✔ CNN-based Face Mask Detection  
    ✔ Eye blink + Head movement Liveness  
    ✔ Covered face / eye warning  
    ✔ Prevents photo & dummy attacks  

    **Developed by:** {DEVELOPER_NAME}  

    **Tech Stack:** Python · TensorFlow · OpenCV · Streamlit
    """)

# ================= FOOTER =================
st.markdown(
    f"<hr><div class='footer'>© 2026 | Developed by <b>{DEVELOPER_NAME}</b></div>",
    unsafe_allow_html=True
)

