import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_assets():
    with open("Notebooks/california_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("Notebooks/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

st.title("California House Predictor")
st.write(f"Powered by XGBoost (R2 Score: 0.7812)")
income = st.slider("Median Income (in 10$ k)",0.5,15.0,8.0)
age = st.slider("House Age",1,52,10)
rooms = st.slider("Total Rooms ",1,1000,100)
bedrooms = st.slider("Total Bedrooms : ",1,6000,500)
population = st.slider("Population : ",3,35000,1400)
household = st.slider("HouseHolds : ",1,6000,500)
latitude = st.slider("Latitude : ",32.5,42.0,34.0)
longitude = st.slider("Longitude : ",-124.3,-114.3,-119.0)

ave_rooms_per_household = rooms / household if household > 0 else 0
ave_bedrooms_per_household = bedrooms / household if household > 0 else 0
ave_occup = population / household if household > 0 else 0

room_per_person = ave_rooms_per_household / ave_occup if ave_occup > 0 else 0
bedroom_ratio = ave_bedrooms_per_household / ave_occup if ave_occup > 0 else 0


input_data = pd.DataFrame([{
    "MedInc": income,
    "HouseAge": age,
    "Population": population,
    "Latitude": latitude,
    "Longitude": longitude,
    "Room_per_person": room_per_person,
    "BedRoom_Ratio": bedroom_ratio
}])

st.write(f"DEBUG - Input Array:")
st.dataframe(input_data)

try:
    scaled_features = scaler.transform(input_data)
    prediction = model.predict(scaled_features)
    final_price = prediction[0] * 100000
    st.success(f"Estimated Price in $ : {final_price:,.2f}")
    st.subheader(f"Raw Prediction : {prediction[0]:.4f}")
    st.divider()
    st.header("Interactive Property Analysis ")
    st.info(f"Currently Analyzing : Latitude {latitude} , Longitude {longitude}")
    map_data = input_data[['Latitude','Longitude']].rename(columns = {
        'Latitude' : 'lat',
        'Longitude' : 'lon'
    })
    st.map(map_data,zoom = 10)
except Exception as e:
    st.error(f"Error making prediction: {e}")
