import streamlit as st

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Rainfall Prediction Portal",
    page_icon="🌧️",
    layout="centered"
)

# --- APP HEADER ---
st.title("🌧️ Rainfall Prediction System")
st.markdown("""
This production-ready web application uses a **Logistic Regression** model trained with **92% accuracy** to predict the likelihood of rainfall based on real-world meteorological observations.
""")
st.markdown("---")

# --- UI LAYOUT DESIGN (Always renders with natural real-world parameters) ---
st.subheader("📊 Enter Current Weather Conditions")
st.write("Provide standard local weather metrics below. Preprocessing and scaling pipelines run automatically on submission:")

# Grid layout with adjusted human-readable ranges and increments based on raw data
col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0, step=0.1, format="%.1f", help="Current local air temperature in Celsius")
    dew = st.number_input("Dew Point (°C)", min_value=-50.0, max_value=40.0, value=12.0, step=0.1, format="%.1f", help="Atmospheric dew point temperature")
    humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=45.0, step=0.1, format="%.1f", help="Percentage of moisture saturation in the air")
    sealevelpressure = st.number_input("Sea Level Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1015.0, step=0.1, format="%.1f", help="Atmospheric pressure adjusted to sea level")

with col2:
    winddir = st.number_input("Wind Direction (Degrees °)", min_value=0.0, max_value=360.0, value=120.0, step=0.1, format="%.1f", help="Compass direction from which the wind originates (0-360°)")
    solarradiation = st.number_input("Solar Radiation (W/m²)", min_value=0.0, max_value=1500.0, value=215.0, step=0.1, format="%.1f", help="Solar power received per unit area")
    windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=250.0, value=15.0, step=0.1, format="%.1f", help="Rate of air movement in kilometers per hour")

st.markdown("---")

# --- ACTION BUTTON ---
predict_btn = st.button("🔮 Predict Rainfall Probability", type="primary", use_container_width=True)

# --- BACKEND LOGIC & DEPENDENCY TRY-EXCEPT (Only executed on click) ---
if predict_btn:
    try:
        # Step 1: Try importing core packages
        import pickle
        import numpy as np
        import pandas as pd

        # Step 2: Try loading assets / pickle files
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scale.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('column.pkl', 'rb') as f:
                model_columns = pickle.load(f)
        except FileNotFoundError as fnf_error:
            st.error(f"❌ **Model Files Missing!** `model.pickle`, `scale.pickle`, or `column.pickle` not found in the root folder.")
            st.info("💡 Please make sure you have generated and uploaded all three pickle files to your GitHub repository.")
            st.stop()  # Safely halt execution

        # Step 3: Parse natural user input features into a DataFrame
        input_data = {
            'temp': temp, 'dew': dew, 'humidity': humidity,
            'sealevelpressure': sealevelpressure, 'winddir': winddir,
            'solarradiation': solarradiation, 'windspeed': windspeed
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Align column indexes matching the training format
        if isinstance(model_columns, list):
            input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Step 4: Scale the raw inputs through scaler object before sending to model
        scaled_features = scaler.transform(input_df)
        
        # Step 5: Execute Model Predictions
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0][1]

        # --- SUCCESSFUL OUTPUT DISPLAY ---
        st.subheader("🎯 Prediction Result")
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            if prediction == 1:
                st.error("🌧️ Prediction: Rain Likely")
            else:
                st.success("☀️ Prediction: No Rain Expected")
                
        with metric_col2:
            st.metric(label="Rainfall Probability", value=f"{prediction_proba * 100:.2f}%")
            
        st.progress(float(prediction_proba))

    except ImportError as imp_error:
        # Display clean layout warning when dependencies are missing
        st.warning("⚠️ **Dependencies Not Installed!**")
        st.error(f"Prediction failed because required processing libraries are missing: `{imp_error.name}`.")
        st.markdown("""
        > 💡 **Developer Note:** To run this predictive model, please install the development dependencies. 
        > Run the following command in your local terminal or update your deployment settings:
        ```bash
        pip install pandas numpy scikit-learn
        ```
        """)
    except Exception as general_error:
        st.error(f"⚠️ An unexpected processing error occurred: {general_error}")