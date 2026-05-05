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
st.title("🚚 Delivery Time Prediction System")
st.markdown("### Smart ML-based delivery time estimator")

st.success("Enter delivery details in sidebar and get instant prediction + insights")

# ---------------- LOAD MODEL ----------------

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
traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
vehicle_map = {"Bike": 0, "Scooter": 1, "Car": 2}
festival_map = {"No": 0, "Yes": 1}
city_map = {"Urban": 0, "Semi-Urban": 1, "Metropolitan": 2}

feature_names = [
    "Age", "Traffic", "Vehicle Condition", "Vehicle Type",
    "Multiple Deliveries", "Festival", "City",
    "Restaurant Prep Time", "Rating", "Distance"
]

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("📦 Delivery Inputs")

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

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 1])

# ---------------- PREDICTION ----------------
with col1:
    st.subheader("📌 Prediction Panel")

    if st.button("🚀 Predict Delivery Time"):

        prediction = model.predict(input_data)[0]

        st.markdown("### ⏱️ Estimated Delivery Time")
        st.success(f"{prediction:.2f} minutes")

        # ---------------- SMART EXPLANATION ----------------
        st.subheader("🧠 Why this prediction?")

        explanation = []

        if distance_km > 10:
            explanation.append("📍 Long distance increases delivery time")

        if Road_traffic_density >= 2:
            explanation.append("🚦 High traffic slows delivery")

        if Time_taken_by_restraunt_in_mins > 40:
            explanation.append("🍽️ Restaurant prep time is high")

        if Delivery_person_Ratings < 3:
            explanation.append("⭐ Low delivery rating affects speed")

        if multiple_deliveries > 1:
            explanation.append("📦 Multiple deliveries increase delay")

        if len(explanation) == 0:
            explanation.append("✔ Conditions are optimal for fast delivery")

        for e in explanation:
            st.write(e)

# ---------------- INPUT SUMMARY ----------------
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

# ---------------- VISUAL INSIGHT ----------------
st.markdown("---")
st.subheader("📈 Feature Impact Visualization")

fig, ax = plt.subplots()

impact_values = [
    Delivery_person_Age * 0.2,
    Road_traffic_density * 2,
    Vehicle_condition * 1,
    Type_of_vehicle * 1.5,
    multiple_deliveries * 2,
    Festival * 2,
    City * 1.5,
    Time_taken_by_restraunt_in_mins * 0.3,
    Delivery_person_Ratings * -1,
    distance_km * 2
]

ax.bar(feature_names, impact_values)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.set_title("Estimated Feature Influence on Delivery Time")

st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 ML Project | Delivery Time Prediction System by Apoorva")
