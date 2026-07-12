# 🌧️ Rainfall Prediction Dashboard using Machine Learning

A production-grade, highly intuitive **Streamlit** web application that predicts the likelihood of rainfall based on climate and atmospheric features. The underlying engine utilizes a robust **Logistic Regression** pipeline, achieving an impressive accuracy metric on test evaluations.

## 🚀 Key Features

- **Production-Ready UI:** Clean, split-column layout with helpful descriptive tooltips for users.
- **Resilient Architecture:** Embedded error handling (`try-except` chains) to capture missing project dependencies or serialization asset errors gracefully without breaking the layout skeleton.
- **Preprocessed Compatibility:** Seamlessly integrates standard scaler mappings directly on user entry streams.

---

## 📊 Model Performance

- **Algorithm:** Logistic Regression
- **Model Accuracy:** `92%`
- **Features Included:** Temperature, Dew Point, Humidity, Sea Level Pressure, Wind Direction, Solar Radiation, and Wind Speed.

---

## 📁 Repository Structure

```text
├── app.py              # Main Streamlit Application UI Code
├── model.pickle        # Trained Logistic Regression Model Weights
├── scale.pickle        # Fitted StandardScaler Instance
├── column.pickle       # Reference list of verified training columns
├── requirements.txt    # Application Dependency Manifest
└── README.md           # Documentation Portal



🌧️ An elegant Streamlit web UI built around a 92% accurate Logistic Regression model to predict rainfall probabilities. Features a resilient UI layout with native error-handling exceptions for dependencies and serialized preprocessing pipelines (Scalers/Models).
```
