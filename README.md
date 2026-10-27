# 🌱 Smart Agriculture

An agriculture support application for crop disease detection and crop yield prediction built with Streamlit, TensorFlow/Keras, and Scikit-Learn.

## 🚀 Features

- **Crop Disease Detection**: Detects crop diseases (e.g., Tomato Healthy vs. Blight) using a fine-tuned MobileNetV2 architecture with calibrated confidence scores and automated leaf validation.
- **Crop Yield Prediction**: Estimates agricultural crop yields based on region, crop type, rainfall, temperature, and pesticide usage.

## 🛠️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/kantalesurya/SmartAgriculture.git
   cd SmartAgriculture
   ```

2. Set up virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```
