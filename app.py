import os
import requests
import streamlit as st
from tensorflow.keras.models import load_model

MODEL_PATH = "mask_detector_model.h5"
MODEL_URL = "https://github.com/rg01818/face-mask-detection-streamlit/releases/download/v1.0/mask_detector_model.h5"

def download_and_validate_model():
    st.info("⬇️ Downloading model… please wait (first run only)")

    r = requests.get(MODEL_URL, stream=True)
    if r.status_code != 200:
        st.error("❌ Model download failed. Check GitHub Release.")
        st.stop()

    # overwrite file every time
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # 🔐 VALIDATION: real model must be large
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if size_mb < 50:
        st.error("❌ Invalid model file downloaded (not a real .h5 model)")
        st.stop()

    st.success(f"✅ Model downloaded successfully ({size_mb:.1f} MB)")

# 🔥 ALWAYS force fresh download on cloud
download_and_validate_model()

model = load_model(MODEL_PATH)
