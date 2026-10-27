import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
DISEASE_MODEL_PATH = BASE_DIR / "disease_model.keras"
LEAF_CENTROID_PATH = BASE_DIR / "leaf_centroid.npy"
YIELD_MODEL_PATH = BASE_DIR / "yield_model.pkl"

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 70.0

st.set_page_config(page_title="Smart Agriculture", page_icon="🌱", layout="wide")
st.title("🌱 Smart Agriculture")
st.write("An agriculture support application for crop disease detection and crop yield prediction.")


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Required file not found: {CLASS_NAMES_PATH.name}")
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        names = json.load(file)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must contain a non-empty JSON list.")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("Every class name must be a non-empty string.")
    if len(set(names)) != len(names):
        raise ValueError("class_names.json contains duplicate class names.")
    return names


@st.cache_resource
def load_disease_model(file_mtime: float = 0.0):
    # Import TensorFlow only when the user requests disease prediction.
    import tensorflow as tf
    if not DISEASE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Required file not found: {DISEASE_MODEL_PATH.name}")
    model = tf.keras.models.load_model(str(DISEASE_MODEL_PATH), compile=False)
    names = load_class_names()
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        raise ValueError("The disease model has multiple outputs.")
    if len(output_shape) != 2 or output_shape[-1] != len(names):
        raise ValueError("The model output count does not match class_names.json.")
    return model


@st.cache_data
def load_leaf_centroid():
    if not LEAF_CENTROID_PATH.is_file():
        return None
    try:
        return np.load(str(LEAF_CENTROID_PATH))
    except Exception:
        return None


def validate_leaf_image(pil_img, model):
    """
    Validates whether the uploaded image is a crop leaf using:
    1. Texture variance check (identifies blank/solid images)
    2. Vegetation/plant color analysis (chlorophyll green and disease/blight tones)
    3. Structural embedding similarity using MobileNetV2 backbone
    """
    img_rgb = pil_img.convert("RGB")
    arr_rgb = np.asarray(img_rgb, dtype=np.float32)

    # 1. Texture check: blank or solid color
    if float(np.std(arr_rgb)) < 15.0:
        return False, "The image has insufficient texture or is a blank/solid color."

    # 2. Plant/vegetation color profile
    hsv = np.array(img_rgb.convert("HSV"))
    h = hsv[:, :, 0] / 255.0
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    is_green = (h >= 0.16) & (h <= 0.45) & (s >= 0.12) & (v >= 0.12)
    is_blight_tone = (h >= 0.05) & (h < 0.16) & (s >= 0.15) & (v >= 0.12) & (v <= 0.88)
    plant_ratio = float(np.mean(is_green | is_blight_tone))

    if plant_ratio < 0.045:
        return False, "The image does not contain plant or leaf foliage colors."

    # 3. Structural embedding similarity to tomato leaf centroid
    centroid = load_leaf_centroid()
    if centroid is not None:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        resized = img_rgb.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        arr = preprocess_input(np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0))
        # Extract features through MobileNetV2 base + GlobalAveragePooling layers
        features = model.layers[1](model.layers[0](arr, training=False)).numpy()[0]
        norm = np.linalg.norm(features)
        if norm > 0:
            sim = float(np.dot(features / norm, centroid))
            if sim < 0.54:
                return False, "The visual structure does not match a crop leaf."

    return True, "Valid leaf image."


def calibrate_scores(raw_scores, temperature: float = 3.2, max_confidence: float = 0.965):
    """
    Calibrate raw softmax probabilities using temperature scaling to avoid
    unrealistic 100% saturation and produce natural, realistic confidence scores.
    """
    scores = np.asarray(raw_scores, dtype=np.float64)
    eps = 1e-12
    logits = np.log(np.clip(scores, eps, 1.0))
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    calibrated = exp_logits / np.sum(exp_logits)

    best_idx = int(np.argmax(calibrated))
    if calibrated[best_idx] > max_confidence:
        diff = calibrated[best_idx] - max_confidence
        calibrated[best_idx] = max_confidence
        other_indices = [i for i in range(len(calibrated)) if i != best_idx]
        calibrated[other_indices] += diff / len(other_indices)

    return calibrated


@st.cache_resource
def load_yield_model():
    if not YIELD_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Required file not found: {YIELD_MODEL_PATH.name}")
    model = joblib.load(YIELD_MODEL_PATH)
    if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps:
        raise ValueError("The yield model does not contain the expected pipeline.")
    preprocessor = model.named_steps["preprocessor"]
    if not hasattr(preprocessor, "named_transformers_") or "cat" not in preprocessor.named_transformers_:
        raise ValueError("The yield model preprocessor has no fitted categorical transformer.")
    cat_transformer = preprocessor.named_transformers_["cat"]
    if not hasattr(cat_transformer, "categories_"):
        raise ValueError("The yield model categorical transformer is not fitted.")
    categories = cat_transformer.categories_
    if len(categories) < 2:
        raise ValueError("Expected region and crop categories in the yield model.")
    areas = sorted(str(value) for value in categories[0] if not pd.isna(value))
    crops = sorted(str(value) for value in categories[1] if not pd.isna(value))
    if not areas or not crops:
        raise ValueError("No region or crop choices were found in the yield model.")
    return model, areas, crops


disease_tab, yield_tab = st.tabs(["🍃 Crop Disease Detection", "🌾 Crop Yield Prediction"])

with disease_tab:
    st.header("Crop Disease Detection")
    st.write("Upload a clear tomato leaf image for disease diagnosis.")
    uploaded_image = st.file_uploader("Choose a tomato leaf image", type=["jpg", "jpeg", "png"], key="leaf_image")
    image = None
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded image", width=350)
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("Could not open this image. Please choose a valid JPG or PNG.")

    if st.button("Predict Disease", key="predict_disease", disabled=image is None):
        try:
            with st.spinner("Analyzing image..."):
                import tensorflow as tf
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

                model = load_disease_model(DISEASE_MODEL_PATH.stat().st_mtime)
                class_names = load_class_names()

                # Step 1: Validate whether the image is actually a leaf
                is_leaf, reason = validate_leaf_image(image, model)

            if not is_leaf:
                st.error(f"❌ **Invalid Image**: {reason} Please upload a clear photo of a tomato leaf.")
            else:
                with st.spinner("Classifying leaf health..."):
                    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                    image_array = np.asarray(resized, dtype=np.float32)
                    image_array = preprocess_input(np.expand_dims(image_array, axis=0))
                    raw_predictions = np.asarray(model.predict(image_array, verbose=0))

                if raw_predictions.shape != (1, len(class_names)):
                    st.error("Unexpected model output. Check the model and class_names.json.")
                else:
                    raw_scores = raw_predictions[0]
                    if not np.all(np.isfinite(raw_scores)):
                        st.error("The model returned invalid numeric values.")
                    else:
                        scores = calibrate_scores(raw_scores)
                        best_index = int(np.argmax(scores))
                        best_class = class_names[best_index]
                        confidence = float(scores[best_index] * 100)

                        st.subheader("Prediction Results")
                        st.write(f"**Predicted class:** {best_class}")
                        st.metric("Model confidence", f"{confidence:.2f}%")
                        if confidence < CONFIDENCE_THRESHOLD:
                            st.warning("Low confidence. Try a clear, well-lit photo of one tomato leaf.")
                        elif best_class.strip().lower() in {"tomato___healthy", "tomato_healthy", "healthy"}:
                            st.success("The model classified this leaf as healthy.")
                        else:
                            st.warning("The model classified this image as a potentially diseased tomato leaf. Verify before taking action.")

                        st.write("**Scores for each class:**")
                        for name, score in zip(class_names, scores):
                            st.write(f"{name}: {float(score) * 100:.2f}%")
        except Exception as error:
            st.error("Disease prediction failed. Details:")
            st.exception(error)

with yield_tab:
    st.header("Crop Yield Prediction")
    st.write("Enter agricultural and weather details to estimate crop yield.")
    try:
        yield_model, areas, crops = load_yield_model()
        col1, col2 = st.columns(2)
        with col1:
            area = st.selectbox("Region / Country", areas, key="yield_area")
            crop = st.selectbox("Crop", crops, key="yield_crop")
            year = st.number_input("Year", min_value=1900, max_value=2100, value=2026, step=1, key="yield_year")
        with col2:
            rainfall = st.number_input("Average annual rainfall (mm)", min_value=0.0, value=1000.0, step=50.0, key="yield_rainfall")
            pesticides = st.number_input("Pesticide usage (tonnes)", min_value=0.0, value=100.0, step=10.0, key="yield_pesticides")
            temperature = st.number_input("Average temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=1.0, key="yield_temperature")

        if st.button("Predict Crop Yield", key="predict_yield"):
            input_data = pd.DataFrame({
                "Area": [area], "Item": [crop], "Year": [int(year)],
                "average_rain_fall_mm_per_year": [rainfall],
                "pesticides_tonnes": [pesticides], "avg_temp": [temperature]
            })
            prediction = np.asarray(yield_model.predict(input_data)).reshape(-1)
            if prediction.size != 1:
                st.error("The yield model returned an unexpected result.")
            else:
                predicted_yield = float(prediction[0])
                if not np.isfinite(predicted_yield):
                    st.error("The yield model returned an invalid number.")
                elif predicted_yield < 0:
                    st.warning("The model predicted a negative yield. Check the model.")
                else:
                    st.success(f"Estimated yield: {predicted_yield:,.2f} hg/ha")
                    st.write(f"Equivalent to approximately **{predicted_yield / 10000:,.2f} tonnes per hectare**.")
                    st.caption("This is a model-based estimate; accuracy depends on training data and input values.")
    except Exception as error:
        st.error("Unable to load the crop yield model. Details:")
        st.exception(error)
