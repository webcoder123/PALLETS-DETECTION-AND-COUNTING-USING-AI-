import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Streamlit page setup
st.set_page_config(page_title="📦 Pallet Detector", layout="centered")

# Header UI
st.markdown("<h1 style='text-align: center;'>📦 Pallet Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to detect and count pallets using YOLOv8.</p>", unsafe_allow_html=True)

# Load YOLOv8 model once
@st.cache_resource
def load_model():
    return YOLO("D:\\360DigiTMG Date 28Aug\\Project_360DigitTMG_HYD_12_04_24_03\\yolov8l.pt")  # Update path if your best.pt is in a different location

model = load_model()

# Upload image
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert to PIL and display original
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Original Image", use_column_width=True)

    with st.spinner("🔍 Detecting pallets..."):
        img_array = np.array(image)
        results = model.predict(img_array, conf=0.5)
        boxes = results[0].boxes
        count = len(boxes) if boxes is not None else 0

        # Annotated image
        annotated = results[0].plot()

    st.image(annotated, caption=f"✅ Detected Pallets: {count}", use_column_width=True)
    st.success(f"🎯 Total Pallets Detected: {count}")

# Footer
st.markdown("---")
st.caption("🚀 Powered by YOLOv8 | Developed with ❤️ using Streamlit")
