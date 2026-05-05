import streamlit as st
import numpy as np
import pickle
import gdown
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Delivery Time Prediction",
    layout="wide",
    page_icon="🚚"
)

# ---------------- HEADER ----------------
st.title("🚚 Delivery Time Prediction App")
st.markdown("### ML-powered system to predict food delivery time")

st.info("Enter details in sidebar → Get prediction + insights + explanation")

# ---------------- MODEL LOAD ----------------

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

# ---------------- FEATURE INFO ----------------
with st.expander("ℹ️ About Model"):
    st.write("""
    - Algorithm: Random Forest Regressor  
    - Purpose: Predict delivery time in minutes  
    - Trained on historical delivery dataset  
    - Inputs include traffic, distance, ratings, etc.
    """)

# ---------------- FEATURE IMPORTANCE (FAKE SAFE VERSION) ----------------
# (Streamlit-safe placeholder since we don’t know model internals exactly)
feature_names = [
    "Age", "Traffic", "Vehicle Condition", "Vehicle Type",
    "Multiple Deliveries", "Festival", "City",
    "Restaurant Time", "Ratings", "Distance"
]

# ---------------- MAPPINGS ----------------
traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
vehicle_map = {"Bike": 0, "Scooter": 1, "Car": 2}
festival_map = {"No": 0, "Yes": 1}
city_map = {"Urban": 0, "Semi-Urban": 1, "Metropolitan": 2}

# ---------------- SIDEBAR ----------------
st.sidebar.header("📦 Input Details")

Delivery_person_Age = st.sidebar.number_input("Age", 18, 65, 25)

Road_traffic_density = traffic_map[
    st.sidebar.selectbox("Traffic Condition", list(traffic_map.keys()))
]

Vehicle_condition = st.sidebar.selectbox("Vehicle Condition", [0, 1, 2])

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
    "Restaurant Prep Time", 5, 120, 20
)

Delivery_person_Ratings = st.sidebar.slider("Rating", 1.0, 5.0, 4.0)

distance_km = st.sidebar.number_input("Distance (km)", 0.1, 50.0, 5.0)

# ---------------- INPUT ARRAY ----------------
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

# ---------------- PREDICTION ----------------
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Predict Delivery Time"):

        prediction = model.predict(input_data)[0]

        st.success(f"⏱️ Estimated Delivery Time: **{prediction:.2f} minutes**")

        # ---------------- SIMPLE EXPLANATION ----------------
        st.subheader("🧠 Why this prediction?")

        if distance_km > 10:
            st.write("📍 Long distance increases delivery time.")
        if Road_traffic_density >= 2:
            st.write("🚦 High traffic slows delivery.")
        if Delivery_person_Ratings < 3:
            st.write("⭐ Lower rating may indicate slower delivery.")
        if Time_taken_by_restraunt_in_mins > 40:
            st.write("🍽️ Restaurant prep time is high.")

        st.write("✔ Model combines all factors to estimate final time.")

# ---------------- INSIGHTS PANEL ----------------
with col2:
    st.subheader("📊 Input Summary")

    df = pd.DataFrame({
        "Feature": feature_names,
        "Value": [
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
        ]
    })

    st.dataframe(df, use_container_width=True)

# ---------------- SIMPLE VISUAL ----------------
st.markdown("---")
st.subheader("📈 Sample Insight Visualization")

fig, ax = plt.subplots()
ax.bar(["Distance", "Traffic", "Rating"], [distance_km, Road_traffic_density, Delivery_person_Ratings])
ax.set_title("Key Input Factors (Scaled View)")
st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with Streamlit | ML Project by Apoorva")
