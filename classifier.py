import streamlit as st
import tensorflow as tf
import numpy as np
import os
import zipfile
import tempfile
from tensorflow.keras.preprocessing import image
from PIL import Image

# =====================================================
# 1️⃣ Load the trained model
# =====================================================
MODEL_PATH = "C:\Users\JOEL\Downloads\archive (6)\alcohol_bottle_classifier.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🍾 Alcohol Bottle Image Classifier")
st.write("Upload your dataset folder (as a ZIP), then upload an image to classify or search for examples.")

# =====================================================
# 2️⃣ Upload dataset ZIP file
# =====================================================
uploaded_zip = st.file_uploader("📦 Upload your dataset ZIP file", type=["zip"])

if uploaded_zip:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    st.success("✅ Dataset extracted successfully!")

    # Detect root folder (in case the ZIP contains a single root directory)
    root_contents = os.listdir(temp_dir)
    if len(root_contents) == 1 and os.path.isdir(os.path.join(temp_dir, root_contents[0])):
        dataset_path = os.path.join(temp_dir, root_contents[0])
    else:
        dataset_path = temp_dir

    # Load class names
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    st.write(f"📚 Found {len(class_names)} classes: {', '.join(class_names)}")

    # =====================================================
    # 3️⃣ Upload image for prediction
    # =====================================================
    st.markdown("---")
    st.subheader("📤 Upload an Image to Classify")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_resized = img.resize((150, 150))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)] if class_names else "Unknown"
        confidence = np.max(predictions)

        st.success(f"### 🏷️ Predicted Class: {predicted_class}")
        st.info(f"Confidence: {confidence*100:.2f}%")

    # =====================================================
    # 4️⃣ Search Functionality
    # =====================================================
    st.markdown("---")
    st.subheader("🔍 Search for Images by Class")

    search_query = st.text_input("Enter class name to search:").strip().lower()

    if search_query:
        matches = [cls for cls in class_names if search_query in cls.lower()]
        if matches:
            for match in matches:
                st.markdown(f"### 📸 {match.capitalize()}")
                folder_path = os.path.join(dataset_path, match)
                images = [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]

                if images:
                    cols = st.columns(3)
                    for idx, img_path in enumerate(images[:9]):  # show first 9
                        with cols[idx % 3]:
                            st.image(img_path, use_column_width=True)
                else:
                    st.warning("No images found in this class folder.")
        else:
            st.error("No matching class found.")
else:
    st.warning("⚠️ Please upload your dataset ZIP file to begin.")

# =====================================================
# 5️⃣ Footer
# =====================================================
st.markdown("---")
st.caption("Developed by Nicolas — Alcohol Bottle Classifier using TensorFlow + Streamlit 🚀")
