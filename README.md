# 🌱 Smart Agriculture — Crop Disease Detection and Yield Prediction

## 1. Project Overview

Smart Agriculture is a machine learning-based application designed to support farmers by identifying tomato leaf diseases and predicting crop yield. The application combines deep learning for image-based disease classification with machine learning for yield estimation.

The system provides a simple, interactive interface built using Streamlit, allowing users to upload tomato leaf images and enter agricultural and weather information to obtain predictions.

## 2. Problem Statement

Farmers often face difficulties identifying crop diseases at an early stage and estimating crop production accurately. Manual disease identification requires agricultural knowledge, while crop yield depends on several factors, including rainfall, temperature, crop type, and pesticide usage.

This project aims to provide an accessible application that helps users identify possible tomato leaf diseases and estimate crop yield using machine learning models.

## 3. Objectives

- Detect healthy and blight-affected tomato leaves from uploaded images.
- Estimate crop yield using agricultural and weather-related information.
- Provide a simple and user-friendly web interface.
- Demonstrate the practical application of deep learning and machine learning in agriculture.
- Support agricultural decision-making through model-based predictions.

## 4. Proposed Solution

The application consists of two main modules.

**Crop Disease Detection:** Users upload a tomato leaf image in JPG, JPEG, or PNG format. A deep learning model processes the image and predicts whether it belongs to the healthy or blight class.

**Crop Yield Prediction:** Users enter the region or country, crop type, year, average annual rainfall, pesticide usage, and average temperature. A trained machine learning model uses these inputs to estimate crop yield.

Both modules are integrated into a single Streamlit application.

## 5. Key Features

- Image upload for tomato leaf disease detection.
- Classification into healthy and blight categories.
- Display of prediction confidence and individual class scores.
- Crop yield estimation using agricultural input parameters.
- Automatic loading of trained models.
- Interactive interface with separate disease detection and yield prediction tabs.
- Input validation and error handling for common application issues.

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| TensorFlow / Keras | Loading and running the deep learning model |
| MobileNetV2 | Transfer learning architecture used for disease classification |
| Scikit-learn | Machine learning pipeline for yield prediction |
| Random Forest Regressor | Crop yield estimation |
| Pandas | Handling agricultural datasets |
| NumPy | Numerical operations and image arrays |
| Pillow (PIL) | Image loading, conversion, and resizing |
| Joblib | Loading the saved yield model |

## 7. Dataset Description

### 7.1 Tomato Leaf Image Dataset

A tomato leaf image dataset is used to train the disease classification model. The images represent the classes used by the trained model.

The current application uses two classes:

- **Tomato___Healthy:** Healthy tomato leaves.
- **Tomato___Blight:** Tomato leaves affected by the blight category represented in the training data.

The dataset images are resized to 224 × 224 pixels before being passed to the model.

### 7.2 Crop Yield Dataset

The crop yield model was trained using the `yield_df.csv` dataset. Related agricultural data files included in the downloaded dataset archive were:

- `yield_df.csv`
- `yield.csv`
- `rainfall.csv`
- `temp.csv`
- `pesticide.csv`

The trained yield model uses the following input features:

- Area or country
- Crop type
- Year
- Average annual rainfall
- Pesticide usage
- Average temperature

The yield dataset contains historical agricultural information used to train the prediction model.

## 8. System Architecture

The project follows a simple workflow:

1. The user opens the Streamlit application.
2. The user selects either Crop Disease Detection or Crop Yield Prediction.
3. The application collects an image or agricultural input values.
4. The corresponding trained model processes the input.
5. The application displays the predicted class or estimated crop yield.

### Disease Detection Workflow

`Tomato Leaf Image → Image Preprocessing → MobileNetV2-Based Model → Class Scores → Predicted Class`

### Yield Prediction Workflow

`Agricultural Inputs → Data Preprocessing → Random Forest Regressor → Estimated Crop Yield`

## 9. Model Details

### 9.1 Crop Disease Detection Model

The disease detection model uses MobileNetV2 transfer learning. It was trained in Google Colab and saved in Keras format.

**Model file:** `disease_model.keras`

**Class labels file:** `class_names.json`

The application resizes uploaded images to 224 × 224 pixels, converts them to numerical arrays, applies the configured preprocessing, and passes them to the trained model.

The model returns scores for the available classes. The class with the highest score is selected as the predicted class.

The application also displays the model's confidence score. Predictions should be verified because incorrect classifications are possible.

### 9.2 Crop Yield Prediction Model

The yield prediction model uses a Scikit-learn Pipeline containing preprocessing and a Random Forest Regressor.

**Model file:** `yield_model.pkl`

The pipeline handles categorical inputs, such as the area and crop type, together with numerical agricultural features.

The application loads the saved pipeline and supplies the user's input values in a Pandas DataFrame. The model returns an estimated crop yield.

The result is displayed in hectograms per hectare (hg/ha), along with an approximate conversion to tonnes per hectare.

## 10. Project Structure

```text
SmartAgriculture/
│
├── app.py
├── disease_model.keras
├── class_names.json
├── yield_model.pkl
├── requirements.txt
└── .venv311/
```

### File Descriptions

- `app.py` — Main Streamlit application containing both prediction modules.
- `disease_model.keras` — Trained deep learning model for tomato leaf classification.
- `class_names.json` — Class labels corresponding to the disease model's output.
- `yield_model.pkl` — Saved machine learning pipeline for crop yield prediction.
- `requirements.txt` — Python dependencies required to run the application.
- `.venv311/` — Local Python virtual environment used to isolate project dependencies.

The original training datasets and Google Colab notebooks are not required to run the application once the trained model files are available.

## 11. Installation and Setup

### Prerequisites

- Python 3.11
- pip package manager
- Terminal
- A web browser
- The trained model files listed in the project structure

### Step 1: Open the Project Directory

Open Terminal and navigate to the project folder.

```bash
cd ~/SmartAgriculture
```

### Step 2: Activate the Virtual Environment

If the virtual environment already exists, activate it using:

```bash
source .venv311/bin/activate
```

### Step 3: Install Dependencies

Install the packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, create it with the required dependencies before running the application.

### Step 4: Run the Application

Start the Streamlit application with:

```bash
python -m streamlit run app.py
```

### Step 5: Open the Application

Open the local URL displayed in Terminal. It is usually:

`http://localhost:8501`

The application should display the Crop Disease Detection and Crop Yield Prediction tabs.

## 12. How to Use the Application

### Crop Disease Detection

1. Open the Crop Disease Detection tab.
2. Upload a clear tomato leaf image in JPG, JPEG, or PNG format.
3. Click **Predict Disease**.
4. View the predicted class, confidence score, and scores for each class.

### Crop Yield Prediction

1. Open the Crop Yield Prediction tab.
2. Select a region or country.
3. Select the crop type.
4. Enter the year, average annual rainfall, pesticide usage, and average temperature.
5. Click **Predict Crop Yield**.
6. View the estimated yield in hg/ha and the approximate equivalent in tonnes per hectare.

## 13. Validation and Error Handling

The application includes checks to improve reliability:

- Verifies that required model and class-label files exist.
- Checks that the number of model output classes matches the class-label list.
- Validates uploaded image formats.
- Checks prediction outputs for invalid numerical values.
- Displays warnings for low-confidence disease predictions.
- Checks yield predictions for unexpected or invalid results.
- Handles model-loading and prediction errors by displaying diagnostic information.

These checks improve application stability but do not guarantee that every prediction is correct.

## 14. Limitations

- Disease detection is limited to the classes represented in the trained model.
- The current model distinguishes healthy leaves from the trained blight category; it does not independently identify early blight and late blight as separate classes.
- Disease classification accuracy depends on the quality of the training dataset, image preprocessing, and model training.
- Poor lighting, blurry images, backgrounds, and unfamiliar leaf conditions may affect predictions.
- The yield model provides estimates based on historical training data and the supplied input values.
- Predictions may be unreliable for inputs outside the training data's range or distribution.
- The application is intended as a decision-support tool, not a replacement for professional agricultural diagnosis.

## 15. Future Enhancements

- Expand disease detection to include additional tomato diseases.
- Improve classification accuracy using more diverse, correctly labelled images.
- Evaluate the disease model using a separate test dataset and a confusion matrix.
- Add additional crops and agricultural datasets.
- Improve yield prediction through model comparison and systematic evaluation.
- Add visual explanations to highlight image regions influencing disease predictions.
- Develop a deployment-ready version accessible to farmers through the internet.
- Add multilingual support and agricultural recommendations validated by domain experts.

## 16. Expected Outcome

The project provides a single application that combines tomato leaf disease classification and crop yield estimation. It demonstrates how deep learning and machine learning can be applied to agricultural problems through an accessible web interface.

The system can assist users in identifying possible tomato leaf diseases and obtaining preliminary yield estimates from agricultural and weather-related inputs.

## 17. Conclusion

Smart Agriculture demonstrates the integration of image classification and regression-based prediction in one practical application. By combining a MobileNetV2-based disease classification model, a Random Forest-based yield prediction pipeline, and a Streamlit interface, the project provides a foundation for data-driven agricultural support.

The reliability of its results depends on the quality of the datasets, the training process, and the consistency between model training and application-time preprocessing. Further model evaluation and improvements can make the system more useful for real-world agricultural applications.

## 18. Project Author

**Project Title:** Smart Agriculture — Crop Disease Detection and Yield Prediction

**Domain:** Machine Learning, Deep Learning, and Smart Agriculture

**Application Framework:** Streamlit

**Development Environment:** Python and Google Colab

---

*Note: Model accuracy values and dataset sizes are not specified here because they should be reported only after they have been measured or verified.*
