import streamlit as st
import numpy as np
import pickle
import gdown
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SwiftDelivery AI",
    page_icon="🚚",
    layout="wide"
)

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

# ---------------- HEADER (STARTUP STYLE) ----------------
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#2E86C1;">🚚 SwiftDelivery AI</h1>
        <h4 style="color:gray;">Predict delivery time instantly using Machine Learning</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- INFO CARDS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("⚡ Fast Prediction", "Real-time")
col2.metric("🧠 AI Model", "Random Forest")
col3.metric("📦 Use Case", "Delivery ETA")

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📦 Order Configuration")

Delivery_person_Age = st.sidebar.number_input("Age", 18, 65, 25)

traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
vehicle_map = {"Bike": 0, "Scooter": 1, "Car": 2}
festival_map = {"No": 0, "Yes": 1}
city_map = {"Urban": 0, "Semi-Urban": 1, "Metropolitan": 2}

Road_traffic_density = traffic_map[
    st.sidebar.selectbox("Traffic Level", list(traffic_map.keys()))
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

Restaurant_time = st.sidebar.number_input("Restaurant Prep Time (min)", 5, 120, 20)

Rating = st.sidebar.slider("Delivery Rating", 1.0, 5.0, 4.0)

distance_km = st.sidebar.number_input("Distance (km)", 0.1, 50.0, 5.0)

# ---------------- INPUT ----------------
input_data = np.array([[
    Delivery_person_Age,
    Road_traffic_density,
    Vehicle_condition,
    Type_of_vehicle,
    multiple_deliveries,
    Festival,
    City,
    Restaurant_time,
    Rating,
    distance_km
]])

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.2, 1])

# ---------------- PREDICTION PANEL ----------------
with left:

    st.subheader("🚀 Prediction Engine")

    if st.button("Predict Delivery Time"):

        prediction = model.predict(input_data)[0]

        st.success(f"⏱️ Estimated Delivery Time: **{prediction:.2f} minutes**")

        st.markdown("### 🧠 Smart Explanation")

        insights = []

        if distance_km > 10:
            insights.append("📍 Long distance increases delivery time")

        if Road_traffic_density >= 2:
            insights.append("🚦 High traffic slows delivery")

        if Restaurant_time > 40:
            insights.append("🍽️ Restaurant preparation delay")

        if Rating < 3:
            insights.append("⭐ Low delivery efficiency rating")

        if multiple_deliveries > 1:
            insights.append("📦 Multiple orders increase delay")

        if len(insights) == 0:
            insights.append("✔ Conditions are optimal for fast delivery")

        for i in insights:
            st.write(i)

# ---------------- DASHBOARD PANEL ----------------
with right:

    st.subheader("📊 Order Summary Dashboard")

    df = pd.DataFrame({
        "Feature": [
            "Age", "Traffic", "Vehicle", "Vehicle Type",
            "Multiple Deliveries", "Festival", "City",
            "Restaurant Time", "Rating", "Distance"
        ],
        "Value": [
            Delivery_person_Age,
            Road_traffic_density,
            Vehicle_condition,
            Type_of_vehicle,
            multiple_deliveries,
            Festival,
            City,
            Restaurant_time,
            Rating,
            distance_km
        ]
    })

    st.dataframe(df, use_container_width=True)

# ---------------- FEATURE IMPACT VISUAL ----------------
st.markdown("---")
st.subheader("📈 AI Feature Influence Overview")

features = [
    "Age", "Traffic", "Vehicle", "Type",
    "Multiple Orders", "Festival", "City",
    "Prep Time", "Rating", "Distance"
]

impact = [
    Delivery_person_Age * 0.1,
    Road_traffic_density * 2,
    Vehicle_condition * 1,
    Type_of_vehicle * 1.2,
    multiple_deliveries * 1.8,
    Festival * 2,
    City * 1.5,
    Restaurant_time * 0.4,
    Rating * -1.2,
    distance_km * 2.2
]

fig, ax = plt.subplots()
ax.barh(features, impact)
ax.set_title("Feature Impact on Delivery Time")
st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 SwiftDelivery AI | Built with Streamlit | ML Project by Apoorva")
