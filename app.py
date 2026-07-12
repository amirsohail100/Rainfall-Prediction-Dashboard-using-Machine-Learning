import streamlit as st
import numpy as np
import pandas as pd

# --- DEPENDENCY & FILE LOADING PROTECTION ---
try:
    import pickle
    
    # Pickle files ko load karne ka try block
    try:
        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)
        with open('scale.pickle', 'rb') as f:
            scaler = pickle.load(f)
        with open('column.pickle', 'rb') as f:
            model_columns = pickle.load(f)
        files_loaded = True
    except FileNotFoundError as e:
        st.error(f"❌ Required model files missing: {e.filename}. Please ensure 'model.pickle', 'scale.pickle', and 'column.pickle' are in the same directory.")
        files_loaded = False
except ImportError as e:
    st.error(f"❌ Missing critical dependencies: {e}. Please check your requirements.txt file.")
    files_loaded = False

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Rainfall Prediction Portal",
    page_icon="🌧️",
    layout="centered"
)

# --- APP HEADER ---
st.title("🌧️ Rainfall Prediction System")
st.markdown("""
This production-ready web application uses a **Logistic Regression** model trained with **92% accuracy** to predict the likelihood of rainfall based on meteorological parameters.
""")
st.markdown("---")

if files_loaded:
    st.subheader("📊 Enter Weather Parameters")
    st.write("Please provide the following environmental metrics (normalized features will be processed automatically):")

    # Creating a clean grid layout for inputs
    col1, col2 = st.columns(2)

    with col1:
        temp = st.number_input("Temperature Component", value=0.0, format="%.6f", help="Standardized temperature value")
        dew = st.number_input("Dew Point", value=0.0, format="%.6f", help="Standardized dew point value")
        humidity = st.number_input("Humidity Level", value=0.0, format="%.6f", help="Standardized humidity level")
        sealevelpressure = st.number_input("Sea Level Pressure", value=0.0, format="%.6f", help="Standardized sea level pressure")

    with col2:
        winddir = st.number_input("Wind Direction Component", value=0.0, format="%.6f", help="Standardized wind direction")
        solarradiation = st.number_input("Solar Radiation", value=0.0, format="%.6f", help="Standardized solar radiation")
        windspeed = st.number_input("Wind Speed", value=0.0, format="%.6f", help="Standardized wind speed")

    # --- PREDICTION LOGIC ---
    st.markdown("---")
    if st.button("🔮 Predict Rainfall Probability", type="primary", use_container_width=True):
        try:
            # 1. Create a dictionary mapping input to columns
            # Using exact names as per your dataset image
            input_data = {
                'temp': temp,
                'dew': dew,
                'humidity': humidity,
                'sealevelpressure': sealevelpressure,
                'winddir': winddir,
                'solarradiation': solarradiation,
                'windspeed': windspeed
            }
            
            # Convert to DataFrame matching training structure
            input_df = pd.DataFrame([input_data])
            
            # Ensure columns order aligns with column.pickle if needed
            if isinstance(model_columns, list):
                input_df = input_df.reindex(columns=model_columns, fill_value=0)

            # 2. Scale the input data using scaled.pickle
            scaled_features = scaler.transform(input_df)

            # 3. Predict using model.pickle
            prediction = model.predict(scaled_features)[0]
            prediction_proba = model.predict_proba(scaled_features)[0][1]

            # --- DISPLAY RESULTS ---
            st.subheader("🎯 Prediction Result")
            
            # Metrics display
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                if prediction == 1:
                    st.error("🌧️ Prediction: Rain Likely")
                else:
                    st.success("☀️ Prediction: No Rain Expected")
                    
            with metric_col2:
                st.metric(label="Rainfall Probability", value=f"{prediction_proba * 100:.2f}%")
                
            # Progress bar for visual aesthetics
            st.progress(float(prediction_proba))

        except Exception as error:
            st.error(f"⚠️ An error occurred during feature processing or prediction: {error}")
            st.info("Please verify that your input shape matches the scalar and model constraints.")
else:
    st.warning("⚠️ Application is currently running in limited mode due to missing core assets.")