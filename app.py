import streamlit as st
import pandas as pd
import numpy as np
import joblib

knn_model = joblib.load("knn_model.pkl")
lr_model = joblib.load("lr_model.pkl")
svr_model = joblib.load("svr_model.pkl")
features = joblib.load("cab_weather_features.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Uber/Lyft Price Predictor", layout="wide")
st.title("🚕 Uber & Lyft Cab Price Predictor")
st.markdown("Predict the fare of your next ride.")

# Defining Input Fields

locations = ['Back Bay', 'Beacon Hill', 'Boston University', 'Fenway', 'Financial District', 
             'Haymarket Square', 'North End', 'North Station', 'Northeastern University', 
             'South Station', 'Theatre District', 'West End']

cab_types = {'Uber': 0, 'Lyft': 1}

cab_names = ['UberPool', 'UberX', 'UberXL', 'Black', 'Black SUV', 'WAV', 
             'Shared', 'Lyft', 'Lyft XL', 'Lux', 'Lux Black', 'Lux Black XL']

st.sidebar.header("Configure Your Ride")
st.subheader("📍 Route Details")

# Creating columns
col1, col2, col3 = st.columns(3)

with col1:
    source = st.selectbox("Pickup Location", locations, index=5)
with col2:
    destination = st.selectbox("Dropoff Location", locations, index=0)
with col3:
    distance = st.number_input("Distance (in miles)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)

col4, col5, col6 = st.columns(3)
with col4:
    cab_type_label = st.selectbox("Company", list(cab_types.keys()))
    cab_type = cab_types[cab_type_label]
with col5:
    name = st.selectbox("Cab Service Type", cab_names)
with col6:
    surge_multiplier = st.selectbox("Surge Multiplier", [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])

st.divider()

st.subheader("🌤️ Environment & Time")
col7, col8, col9 = st.columns(3)

with col7:
    rain_category = st.selectbox("Precipitation", ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain'])
    temp = st.number_input("Temperature (°F)", min_value=-10.0, max_value=100.0, value=40.0)
with col8:
    pressure = st.number_input("Pressure (mb)", min_value=900.0, max_value=1100.0, value=1010.0)
    wind = st.number_input("Wind Speed (mph)", min_value=0.0, max_value=50.0, value=5.0)
with col9:
    day_part = st.selectbox("Time of Day", ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night'])
    day_type_label = st.radio("Day of Week", ['Weekday', 'Weekend'])
    dayname = 0 if day_type_label == 'Weekend' else 1

# Prediction
if st.button("Calculate Estimated Fare", type="primary", use_container_width=True):
    input_data = {
        'distance': distance,
        'cab_type': cab_type,
        'surge_multiplier': surge_multiplier,
        'temp': temp,
        'clouds': 0.5, # Assuming a static average if not collected, or you can add to UI
        'pressure': pressure,
        'rain': 0.0,   # Raw rain amount isn't needed if rain_category is used, but needs to be in df for scaler
        'humidity': 0.7, # Assuming static average
        'wind': wind,
        'dayname': dayname
    }

    df_input = pd.DataFrame([input_data])

    cols_to_scale = ['distance', 'surge_multiplier', 'temp', 'pressure', 'wind']
    df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])

    df_final = pd.DataFrame(0, index=np.arange(1), columns=features)

    for col in df_input.columns:
        if col in df_final.columns:
            df_final[col] = df_input[col]

    categorical_mappings = [
        f"destination_{destination}",
        f"location_{source}",
        f"name_{name}",
        f"rain_category_{rain_category}",
        f"day_part_{day_part}"
    ]

    for mapping in categorical_mappings:
        if mapping in df_final.columns:
            df_final[mapping] = 1

    pred_knn = knn_model.predict(df_final)[0]
    pred_lr = lr_model.predict(df_final)[0]
    pred_svr = svr_model.predict(df_final)[0]

    prediction = np.mean([pred_knn, pred_lr, pred_svr])
    
    st.success(f"### Estimated Fare: ${prediction:.2f}")
    st.markdown(f"**Model Breakdown:** KNN: ${pred_knn:.2f} | LR: ${pred_lr:.2f} | SVR: ${pred_svr:.2f}")