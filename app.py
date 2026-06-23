
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏢",
    layout="wide"
)

# --------------------------
# LOAD FILES
# --------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------
# CUSTOM CSS
# --------------------------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.hero {
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}

.result-card {
    background: linear-gradient(135deg,#10b981,#059669);
    padding: 25px;
    border-radius: 20px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# HERO SECTION
# --------------------------
st.markdown("""
<div class="hero">
<h1>🏢 Real Estate Price Predictor</h1>
<p>Modern ML Dashboard for House Price Prediction</p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# TOP CARDS
# --------------------------
c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown("<div class='metric-card'><h3>Linear Regression</h3><p>Model</p></div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='metric-card'><h3>16</h3><p>Features</p></div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='metric-card'><h3>Feature Engineering</h3><p>Enabled</p></div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='metric-card'><h3>ML Powered</h3><p>Prediction</p></div>", unsafe_allow_html=True)

st.write("")

# --------------------------
# INPUT SECTION
# --------------------------
left,right = st.columns(2)

with left:

    area = st.number_input("📏 Area (sq ft)", value=5000)

    bedrooms = st.number_input(
        "🛏 Bedrooms",
        min_value=1,
        value=3
    )

    bathrooms = st.number_input(
        "🚿 Bathrooms",
        min_value=1,
        value=2
    )

    stories = st.number_input(
        "🏠 Stories",
        min_value=1,
        value=2
    )

    parking = st.number_input(
        "🚗 Parking",
        min_value=0,
        value=1
    )

with right:

    mainroad = st.selectbox(
        "Main Road Access",
        ["No","Yes"]
    )

    guestroom = st.selectbox(
        "Guest Room",
        ["No","Yes"]
    )

    basement = st.selectbox(
        "Basement",
        ["No","Yes"]
    )

    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["No","Yes"]
    )

    airconditioning = st.selectbox(
        "Air Conditioning",
        ["No","Yes"]
    )

    prefarea = st.selectbox(
        "Preferred Area",
        ["No","Yes"]
    )

    furnishing = st.selectbox(
        "Furnishing Status",
        [
            "Furnished",
            "Semi-Furnished",
            "Unfurnished"
        ]
    )

# --------------------------
# HELPERS
# --------------------------
def yn(x):
    return 1 if x=="Yes" else 0

# --------------------------
# PREDICT
# --------------------------
if st.button("🚀 Predict House Price", use_container_width=True):

    total_rooms = bedrooms + bathrooms

    area_per_room = (
        area / total_rooms
        if total_rooms != 0
        else 0
    )

    area_story = area * stories

    semi = 1 if furnishing=="Semi-Furnished" else 0
    unfurnished = 1 if furnishing=="Unfurnished" else 0

    features = pd.DataFrame([[
        area,
        bedrooms,
        bathrooms,
        stories,
        yn(mainroad),
        yn(guestroom),
        yn(basement),
        yn(hotwaterheating),
        yn(airconditioning),
        parking,
        yn(prefarea),
        semi,
        unfurnished,
        total_rooms,
        area_per_room,
        area_story
    ]], columns=[
        'area',
        'bedrooms',
        'bathrooms',
        'stories',
        'mainroad',
        'guestroom',
        'basement',
        'hotwaterheating',
        'airconditioning',
        'parking',
        'prefarea',
        'furnishingstatus_semi-furnished',
        'furnishingstatus_unfurnished',
        'total_rooms',
        'area_per_room',
        'area_story'
    ])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]

    st.balloons()

    st.markdown(
        f"""
        <div class="result-card">
        <h2>💰 Estimated House Price</h2>
        <h1>₹ {prediction:,.0f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric(
        "Predicted Value",
        f"₹ {prediction:,.0f}"
    )

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.caption(
    "Built with ❤️ using Streamlit, Scikit-Learn and Linear Regression"
)
