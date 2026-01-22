import streamlit as st
import tensorflow as tf
import numpy as np
import os
import zipfile
import tempfile
from tensorflow.keras.preprocessing import image
from PIL import Image

st.title("🍾 Alcohol Bottle Classifier")
st.write("Upload your trained model, your dataset ZIP, then classify images or search classes.")

# =====================================================
# 1️⃣ Upload the trained model (.h5)
# =====================================================
uploaded_model = st.file_uploader("Upload your trained model (.h5)", type=["h5"])
if uploaded_model:
    # Save uploaded model to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model_file:
        tmp_model_file.write(uploaded_model.read())
        tmp_model_path = tmp_model_file.name

    # Load the model
    model = tf.keras.models.load_model(tmp_model_path)
    st.success("✅ Model loaded successfully!")

    # =====================================================
    # 2️⃣ Upload dataset ZIP file
    # =====================================================
    uploaded_zip = st.file_uploader("Upload your dataset ZIP file", type=["zip"])
    if uploaded_zip:
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        st.success("✅ Dataset extracted successfully!")

        # Detect root folder (if ZIP contains a single root directory)
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

        # Check for class mismatch
        model_output_size = model.output_shape[-1]  # number of units in final layer
        if model_output_size != len(class_names):
            st.warning(f"⚠️ Model output ({model_output_size}) does not match dataset classes ({len(class_names)}). Classification may be incorrect.")
            class_mismatch = True
        else:
            class_mismatch = False

        # =====================================================
        # 3️⃣ Upload image for prediction
        # =====================================================
        st.markdown("---")
        st.subheader("📤 Upload an Image to Classify")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="img_upload")

        if uploaded_image:
            img = Image.open(uploaded_image).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess
            img_resized = img.resize((150, 150))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            predictions = model.predict(img_array)
            predictions = np.squeeze(predictions)

            if not class_mismatch and predictions.size == len(class_names):
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions)
            else:
                predicted_class = "Unknown (class mismatch)"
                confidence = 0.0

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
        st.warning("⚠️ Please upload your dataset ZIP file to continue.")
else:
    st.warning("⚠️ Please upload your trained `.h5` model to start.")

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.caption("Developed by Nicolas — Alcohol Bottle Classifier using TensorFlow + Streamlit 🚀")
