import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. LOAD TRAINED MODELS & SCALER
# ==========================================
@st.cache_resource
def load_models():
    reg_model = joblib.load("reg_model.pkl")   # Regression model (Total_Trip_Cost)
    clf_model = joblib.load("clf_model.pkl")   # Classification model (Cost_Category)
    scaler = joblib.load("scaler.pkl")         # StandardScaler
    return reg_model, clf_model, scaler

reg_model, clf_model, scaler = load_models()

# ==========================================
# 2. FEATURE COLUMNS (MUST MATCH TRAINING)
# ==========================================
feature_cols = [
    'City',
    'Season',
    'Trip_Days',
    'Flight_Type',
    'Distance_km',
    'Hotel_Class',
    'Daily_Hotel_Cost',
    'Daily_Food_Cost',
    'Transport_Mode',
    'Daily_Transport_Cost',
    'Activities_Count',
    'Shopping_Cost'
]

# ---- City ----
city_options = ["London", "Dubai", "Goa", "Bangkok", "Kaula Lumpur", "Istanbul"]  
city_to_code = {
    "London": 0,   
    "Dubai": 1,
    "Goa": 2,
    "Bangkok": 3,
    "Kaula Lumpur": 4,
    "Istanbul": 5
}

# ---- Season ----
season_options = ["Peak", "Winter", "Monsoon", "Holiday","Off-Season"]  
season_to_code = {
    "Peak": 0,  
    "Winter": 1,
    "Monsoon": 2,
    "Holiday": 3,
    "Off-Season": 4
}

# ---- Flight Type ----
flight_options = ["Economy","Premium Economy", "Business", "First"]  
flight_to_code = {
    "Economy": 0,   
    "Premium Economy": 1,
    "Business": 2,
    "First": 3
}

# ---- Hotel Class ----
hotel_options = ["Budget", "Standard", "Luxury", "Premium"]  
hotel_to_code = {
    "Budget": 0,  
    "Standard": 1,
    "Luxury": 2,
    "Premium": 3,
}

# ---- Transport Mode ----
transport_options = ["Private Car", "Public", "Rideshare", "Taxi"]  
transport_to_code = {
    "Private Car": 0,      
    "Public": 1,
    "Rideshare": 2,
    "Taxi": 3
}

# ---- Cost Category ----
cost_code_to_label = {
    0: "Budget",      
    1: "Luxury",
    2: "Mid-range"
}

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Travel Cost Prediction", page_icon="✈️")
st.title("✈️ Intelligent Travel Cost Predictor")
st.write("Fill your trip details to predict **Total Trip Cost** and **Cost Category**.")

st.sidebar.header("🧾 Trip Details")

City_str = st.sidebar.selectbox("City", city_options)
Season_str = st.sidebar.selectbox("Season", season_options)
Flight_Type_str = st.sidebar.selectbox("Flight Type", flight_options)
Hotel_Class_str = st.sidebar.selectbox("Hotel Class", hotel_options)
Transport_Mode_str = st.sidebar.selectbox("Transport Mode", transport_options)

Trip_Days = st.sidebar.number_input("Trip Days", min_value=1, step=1, value=1)
Distance_km = st.sidebar.number_input("Distance (km)", min_value=0.0, step=10.0)
Daily_Hotel_Cost = st.sidebar.number_input("Daily Hotel Cost (₹)", min_value=0.0, step=100.0)
Daily_Food_Cost = st.sidebar.number_input("Daily Food Cost (₹)", min_value=0.0, step=100.0)
Daily_Transport_Cost = st.sidebar.number_input("Daily Transport Cost (₹)", min_value=0.0, step=50.0)
Activities_Count = st.sidebar.number_input("Activities Count", min_value=0, step=1, value=3)
Shopping_Cost = st.sidebar.number_input("Shopping Cost (₹)", min_value=0.0, step=500.0)

# ==========================================
# 5. CONVERT NAMES → CODES (FOR MODEL)
# ==========================================
City = city_to_code[City_str]
Season = season_to_code[Season_str]
Flight_Type = flight_to_code[Flight_Type_str]
Hotel_Class = hotel_to_code[Hotel_Class_str]
Transport_Mode = transport_to_code[Transport_Mode_str]

# Build dataframe in the same order as feature_cols
user_df = pd.DataFrame([[
    City,
    Season,
    Trip_Days,
    Flight_Type,
    Distance_km,
    Hotel_Class,
    Daily_Hotel_Cost,
    Daily_Food_Cost,
    Transport_Mode,
    Daily_Transport_Cost,
    Activities_Count,
    Shopping_Cost
]], columns=feature_cols)


# Scale with the same scaler used in training
user_scaled = scaler.transform(user_df)

# ==========================================
# 6. PREDICTION
# ==========================================
task = st.selectbox("What do you want to predict?", ["Total Trip Cost", "Cost Category", "Both"])

if st.button("Predict"):

    # Regression → Total Trip Cost
    if task in ["Total Trip Cost", "Both"]:
        reg_pred = reg_model.predict(user_scaled)[0]
        st.subheader("🔢 Your Total Cost")
        st.write(f"**Predicted Total Trip Cost:** ₹ {reg_pred:,.2f}")

    # Classification → Cost Category
    if task in ["Cost Category", "Both"]:
        clf_pred_code = clf_model.predict(user_scaled)[0]  # e.g., 0/1/2
        clf_label = cost_code_to_label.get(clf_pred_code, f" {clf_pred_code}")
        st.subheader("🏷 Expense Level")
        st.write(f"**Predicted Cost Category:** {clf_label}")
