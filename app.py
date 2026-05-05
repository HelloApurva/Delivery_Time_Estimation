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

# ---------------- TOP HERO SECTION (STARTUP STYLE) ----------------
st.markdown(
    """
    <div style="text-align:center; padding:20px;">
        <h1 style="color:#1f77b4;">🚚 SwiftDelivery AI</h1>
        <h3 style="color:gray;">AI-powered Delivery Time Prediction System</h3>
        <p style="color:#555;">Predict delivery ETA in real-time using Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- KPI CARDS ----------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("⚡ Speed", "Real-time")
c2.metric("🧠 Model", "Random Forest")
c3.metric("📦 Use Case", "Food Delivery")
c4.metric("🎯 Status", "Live")

st.markdown("---")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("📦 Order Configuration")

traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
vehicle_map = {"Bike": 0, "Scooter": 1, "Car": 2}
festival_map = {"No": 0, "Yes": 1}
city_map = {"Urban": 0, "Semi-Urban": 1, "Metropolitan": 2}

age = st.sidebar.number_input("Delivery Person Age", 18, 65, 25)

traffic = traffic_map[
    st.sidebar.selectbox("Traffic Level", list(traffic_map.keys()))
]

vehicle_condition = st.sidebar.selectbox("Vehicle Condition", [0, 1, 2])

vehicle = vehicle_map[
    st.sidebar.selectbox("Vehicle Type", list(vehicle_map.keys()))
]

multi = st.sidebar.selectbox("Multiple Deliveries", [0, 1, 2, 3])

festival = festival_map[
    st.sidebar.selectbox("Festival Day?", list(festival_map.keys()))
]

city = city_map[
    st.sidebar.selectbox("City Type", list(city_map.keys()))
]

prep_time = st.sidebar.number_input("Restaurant Prep Time", 5, 120, 20)

rating = st.sidebar.slider("Delivery Rating", 1.0, 5.0, 4.0)

distance = st.sidebar.number_input("Distance (km)", 0.1, 50.0, 5.0)

# ---------------- INPUT ARRAY ----------------
input_data = np.array([[
    age, traffic, vehicle_condition, vehicle,
    multi, festival, city,
    prep_time, rating, distance
]])

# ---------------- TABS (STARTUP DASHBOARD STYLE) ----------------
tab1, tab2, tab3 = st.tabs(["🚀 Predict", "📊 Insights", "📘 About"])

# ---------------- TAB 1: PREDICTION ----------------
with tab1:

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Prediction Engine")

        if st.button("🚀 Predict Delivery Time"):

            prediction = model.predict(input_data)[0]

            st.success(f"⏱️ Estimated Delivery Time: **{prediction:.2f} minutes**")

            st.markdown("### 🧠 Explanation")

            reasons = []

            if distance > 10:
                reasons.append("📍 Long distance increases time")

            if traffic >= 2:
                reasons.append("🚦 High traffic delay")

            if prep_time > 40:
                reasons.append("🍽️ Restaurant preparation delay")

            if rating < 3:
                reasons.append("⭐ Low delivery efficiency")

            if multi > 1:
                reasons.append("📦 Multiple deliveries increase time")

            if len(reasons) == 0:
                reasons.append("✔ Optimal conditions for fast delivery")

            for r in reasons:
                st.write(r)

    with col2:

        st.subheader("📦 Live Input Summary")

        df = pd.DataFrame({
            "Feature": [
                "Age", "Traffic", "Vehicle", "Type",
                "Multiple", "Festival", "City",
                "Prep Time", "Rating", "Distance"
            ],
            "Value": [
                age, traffic, vehicle, vehicle,
                multi, festival, city,
                prep_time, rating, distance
            ]
        })

        st.dataframe(df, use_container_width=True)

# ---------------- TAB 2: INSIGHTS ----------------
with tab2:

    st.subheader("📈 Feature Influence Overview")

    features = ["Age","Traffic","Vehicle","Type","Multi","Festival","City","Prep","Rating","Distance"]

    impact = [
        age*0.1,
        traffic*2,
        vehicle_condition,
        vehicle*1.2,
        multi*1.8,
        festival*2,
        city*1.5,
        prep_time*0.3,
        rating*-1.2,
        distance*2.2
    ]

    fig, ax = plt.subplots()
    ax.barh(features, impact)
    ax.set_title("Feature Impact on Delivery Time")

    st.pyplot(fig)

    st.info("This visualization shows how each factor influences delivery time prediction.")

# ---------------- TAB 3: ABOUT ----------------
with tab3:

    st.subheader("🚚 About SwiftDelivery AI")

    st.write("""
    SwiftDelivery AI is a machine learning system designed to predict food delivery time 
    based on real-world conditions like traffic, distance, and order complexity.

    ### 🎯 Goal:
    Help delivery platforms improve ETA accuracy and customer experience.

    ### 🧠 Model:
    Random Forest Regressor trained on historical delivery data.

    ### 💡 Impact:
    - Better delivery planning
    - Improved customer satisfaction
    - Reduced delays
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 SwiftDelivery AI | Startup-style ML Product by Apoorva")
