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
This production-ready web application uses a **Logistic Regression** model trained with **92% accuracy** to predict the likelihood of rainfall based on meteorological parameters.
""")
st.markdown("---")

# --- UI LAYOUT DESIGN (Always renders, even if dependencies are missing) ---
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

st.markdown("---")

# --- ACTION BUTTON (Visible to everyone) ---
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
            with open('model.pickle', 'rb') as f:
                model = pickle.load(f)
            with open('scale.pickle', 'rb') as f:
                scaler = pickle.load(f)
            with open('column.pickle', 'rb') as f:
                model_columns = pickle.load(f)
        except FileNotFoundError as fnf_error:
            st.error(f"❌ **Model Files Missing!** `model.pickle`, `scale.pickle`, or `column.pickle` not found in the root folder.")
            st.info("💡 Please make sure you have generated and uploaded all three pickle files to your GitHub repository.")
            st.stop()  # Safely halt execution

        # Step 3: Parse input features into a DataFrame
        input_data = {
            'temp': temp, 'dew': dew, 'humidity': humidity,
            'sealevelpressure': sealevelpressure, 'winddir': winddir,
            'solarradiation': solarradiation, 'windspeed': windspeed
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Align column indexes matching the training format
        if isinstance(model_columns, list):
            input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Step 4: Scale and Predict operations
        scaled_features = scaler.transform(input_df)
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