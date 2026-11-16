import streamlit as st
import numpy as np
from PIL import Image
import joblib

st.set_page_config(page_title="Skin Cancer CNN Demo", page_icon="🔬")
st.title("Skin Cancer Image Classifier — DEMO")
st.error(
    "FAKE / SIMULATED DEMO ONLY. This is NOT a real CNN, has NOT been trained "
    "on real skin-cancer images, and cannot detect or diagnose cancer. "
    "Never use its output for medical decisions."
)

@st.cache_resource
def load_demo_model():
    return joblib.load("skin_cancer_cnn.keras")

model = load_demo_model()
st.write("Upload any JPG or PNG image to demonstrate the app interface.")
file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Run demo prediction"):
        # Deliberately artificial prediction: based only on average brightness.
        arr = np.asarray(image.resize((32, 32)), dtype=np.float32) / 255.0
        brightness = float(arr.mean())
        probs = model.predict_proba(np.array([[brightness]], dtype=np.float32))[0]
        labels = model.demo_class_names
        idx = int(np.argmax(probs))
        st.subheader("Simulated output (not a medical result)")
        st.write("Demo category:", labels[idx])
        st.write(f"Demo score: {probs[idx] * 100:.1f}%")
        st.caption("This score is fabricated by a toy placeholder classifier.")
