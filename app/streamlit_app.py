import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import st_folium
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path

# Load model directly (for Streamlit Cloud deployment)
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[1] / "artifacts" / "california_price_pipeline.joblib"
    return joblib.load(model_path)

# === Page config must be FIRST ===
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# === Initialize session state for prediction ===
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# === Custom transformers (kept only for input UI consistency) ===
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attibute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attibute_names].values

# === Header ===
st.markdown("""
<div style="text-align:center;">
    <h1>🏡 California Housing Price Predictor</h1>
    <p>Enter housing features below to estimate property value</p>
</div>
""", unsafe_allow_html=True)

# === Create Tabs ===
tab1, tab2 = st.tabs(["🏡 Predict Price", "🌍 Map View"])

# === TAB 1: Prediction ===
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input("Longitude", value=-122.23, key="long")
        latitude = st.number_input("Latitude", value=37.88, key="lat")
        housing_median_age = st.number_input("Housing Median Age", value=41.0)
        total_rooms = st.number_input("Total Rooms", value=880.0)

    with col2:
        total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
        population = st.number_input("Population", value=322.0)
        households = st.number_input("Households", value=126.0)
        median_income = st.number_input("Median Income", value=8.3252)
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
            index=0
        )

    # === Predict using loaded model ===
    if st.button("🔮 Predict Price", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }])

        try:
            # Load model and make prediction
            pipeline = load_model()
            prediction = pipeline.predict(input_data)[0]
            
            st.session_state.prediction = prediction
            st.success(f"### 💰 Estimated Median House Value: **${prediction:,.2f}**")

        except Exception as e:
            st.error(f"🚨 Prediction Error: {e}")
            st.info("💡 Make sure the model file exists in the artifacts/ directory")

    # === Neighborhood Stats Section ===
    if st.session_state.prediction:
        st.write("### 📊 Neighborhood Facts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Median Income",
                value=f"${median_income * 10000:,.0f}",
                delta="Regional data"
            )
        
        with col2:
            st.metric(
                label="Population Density",
                value=f"{population/households:.1f}",
                delta="People per household"
            )
        
        with col3:
            st.metric(
                label="Avg Rooms",
                value=f"{total_rooms/households:.1f}",
                delta="Per household"
            )
        
        st.info("""
        📍 **Coming Soon:**  
        • Education Score (Census API)  
        • Crime Rate Index  
        • School District Ratings  
        • Walkability Score  
        """)
    else:
        st.info("💡 Predict price first to reveal neighborhood insights.")

# === TAB 2: Map View ===
with tab2:
    st.write("### 🌍 Property Location")
    
    # Create map centered on the property location
    m = folium.Map(
        location=[latitude, longitude], 
        zoom_start=12,
        tiles="OpenStreetMap"
    )
    
    # Add marker with popup
    popup_text = f"Predicted Price: ${st.session_state.prediction:,.0f}" if st.session_state.prediction else "House Location"
    
    folium.Marker(
        location=[latitude, longitude],
        popup=folium.Popup(popup_text, max_width=200),
        tooltip="Click for details",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(m)
    
    # Add a circle to show approximate neighborhood
    folium.Circle(
        location=[latitude, longitude],
        radius=500,  # 500 meters
        color="blue",
        fill=True,
        fillColor="blue",
        fillOpacity=0.1,
        popup="Neighborhood area"
    ).add_to(m)
    
    # Display map
    st_folium(m, width=900, height=500)
    
    # Property details
    st.write("### 📍 Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"""
        **Location:**  
        - Latitude: {latitude}  
        - Longitude: {longitude}  
        - Ocean Proximity: {ocean_proximity}
        """)
    
    with col2:
        st.write(f"""
        **Property Info:**  
        - Median Age: {housing_median_age} years  
        - Total Rooms: {total_rooms:,.0f}  
        - Households: {households:,.0f}
        """)
    
    if st.session_state.prediction:
        st.success(f"### 💰 Estimated Value: **${st.session_state.prediction:,.2f}**")

