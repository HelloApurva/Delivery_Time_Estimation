import streamlit as st
import numpy as np
import pickle
import gdown

st.set_page_config(page_title="Delivery Time Prediction", layout="centered")

st.title("🚚 Delivery Time Prediction App")
st.write("Enter order details to predict delivery time")

# ---------------- LOAD MODEL FROM GOOGLE DRIVE ----------------

@st.cache_resource
def load_model():

    file_id = "1elMn67oDYcy2OYGS4ynYIcxry4lRPBuD"
    url = f"https://drive.google.com/uc?id={file_id}"

    output = "model.pkl"

    gdown.download(url, output, quiet=False)

    with open(output, "rb") as f:
        model = pickle.load(f)

    return model


model = load_model()

# ---------------- MAPPINGS ----------------

traffic_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Jam": 3
}

vehicle_map = {
    "Bike": 0,
    "Scooter": 1,
    "Car": 2
}

order_map = {
    "Snack": 0,
    "Meal": 1,
    "Drinks": 2,
    "Buffet": 3
}

festival_map = {
    "No": 0,
    "Yes": 1
}

city_map = {
    "Urban": 0,
    "Semi-Urban": 1,
    "Metropolitan": 2
}

# ---------------- INPUTS ----------------

st.sidebar.header("📦 Delivery Details")

Delivery_person_Age = st.sidebar.number_input("Delivery Person Age", 18, 65, 25)

Road_traffic_density = traffic_map[
    st.sidebar.selectbox("Traffic Condition", list(traffic_map.keys()))
]

Vehicle_condition = st.sidebar.selectbox(
    "Vehicle Condition (0 = worst, 2 = best)", [0, 1, 2]
)

Type_of_vehicle = vehicle_map[
    st.sidebar.selectbox("Vehicle Type", list(vehicle_map.keys()))
]

multiple_deliveries = st.sidebar.selectbox("Multiple Deliveries", [0, 1, 2, 3])

Festival = festival_map[
    st.sidebar.selectbox("Festival Day?", list(festival_map.keys()))
]

City = city_map[
    st.sidebar.selectbox("City Type", list(city_map.keys()))
]

Time_taken_by_restraunt_in_mins = st.sidebar.number_input(
    "Restaurant Preparation Time (mins)", 5, 120, 20
)

Delivery_person_Ratings = st.sidebar.slider(
    "Delivery Person Ratings", 1.0, 5.0, 4.0
)

distance_km = st.sidebar.number_input("Distance (km)", 0.1, 50.0, 5.0)

# ---------------- PREDICTION ----------------

if st.button("Predict Delivery Time"):

    input_data = np.array([[
        Delivery_person_Age,
        Road_traffic_density,
        Vehicle_condition,
        Type_of_vehicle,
        multiple_deliveries,
        Festival,
        City,
        Time_taken_by_restraunt_in_mins,
        Delivery_person_Ratings,
        distance_km
    ]])

    prediction = model.predict(input_data)

    st.success(f"⏱️ Estimated Delivery Time: {prediction[0]:.2f} minutes")

st.caption("🚀 ML-powered Delivery Time Prediction System")
