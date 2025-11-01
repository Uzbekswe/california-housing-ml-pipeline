import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import st_folium
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile

# Load model directly (for Streamlit Cloud deployment)
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[1] / "artifacts" / "california_price_pipeline.joblib"
    return joblib.load(model_path)

# Load SHAP explainer
@st.cache_resource
def load_explainer():
    pipeline = load_model()
    # Get the model from the pipeline (last step)
    model = pipeline.named_steps['model']
    return shap.TreeExplainer(model)

# === Page config must be FIRST ===
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# === Custom CSS Styling ===
st.markdown("""
<style>
.css-18e3th9 {
    padding-top: 2rem;
}
h1, h2, h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# === Sidebar Info Panel ===
st.sidebar.title("ℹ️ About This App")
st.sidebar.write("""
Real-estate ML model trained on California Housing Dataset.

**Features:**
- ML model with engineered features
- Map visualization
- Explainable AI (SHAP)
- PDF export
""")

# === Initialize session state for prediction ===
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'prepared_data' not in st.session_state:
    st.session_state.prepared_data = None

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
tab1, tab2, tab3 = st.tabs(["🏡 Predict Price", "🌍 Map View", "🔍 Explain Prediction"])

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
    if st.button("� Predict Price", type="primary"):
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
            
            # Store prepared data for SHAP (transform through preprocessing only)
            prepared_data = pipeline.named_steps['preprocessing'].transform(input_data)
            
            prediction = pipeline.predict(input_data)[0]
            
            # Store in session state
            st.session_state.prediction = prediction
            st.session_state.input_data = input_data
            st.session_state.prepared_data = prepared_data
            
            st.success(f"### 💰 Estimated Median House Value: **${prediction:,.2f}**")
            
            # === PDF Download Button ===
            if st.button("📄 Download Prediction Report as PDF"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    c = canvas.Canvas(tmp.name, pagesize=letter)

                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(30, 750, "California Housing Price Prediction Report")
                    
                    c.setFont("Helvetica", 10)
                    c.drawString(30, 730, "Generated by ML Pipeline")

                    c.setFont("Helvetica", 12)
                    y = 690
                    fields = {
                        "Longitude": longitude,
                        "Latitude": latitude,
                        "Housing Median Age": housing_median_age,
                        "Total Rooms": total_rooms,
                        "Total Bedrooms": total_bedrooms,
                        "Population": population,
                        "Households": households,
                        "Median Income": median_income,
                        "Ocean Proximity": ocean_proximity,
                        "Predicted House Value": f"${prediction:,.2f}",
                    }

                    for key, value in fields.items():
                        c.drawString(30, y, f"{key}: {value}")
                        y -= 25

                    c.showPage()
                    c.save()

                    st.download_button(
                        label="⬇️ Download PDF",
                        data=open(tmp.name, "rb").read(),
                        file_name="housing_prediction_report.pdf",
                        mime="application/pdf"
                    )

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
    st.write("### 🗺 Property Location")
    
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

# === TAB 3: SHAP Explainability ===
with tab3:
    st.write("### 🔍 Model Explainability for Last Prediction")
    
    if st.session_state.prediction is None:
        st.info("💡 Run a prediction first in the '🏡 Predict Price' tab.")
    else:
        st.success(f"### 💡 Predicted Price: **${st.session_state.prediction:,.2f}**")
        
        try:
            # Load explainer
            explainer = load_explainer()
            
            # Compute SHAP values for the prepared data
            shap_values = explainer.shap_values(st.session_state.prepared_data)
            
            st.write("#### 🎯 Feature Contributions")
            st.write("Positive values (red) increase the predicted price, negative values (blue) decrease it.")
            
            # Force plot
            st.write("##### Individual Prediction Breakdown")
            fig, ax = plt.subplots(figsize=(12, 3))
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                st.session_state.prepared_data[0],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig, bbox_inches='tight')
            plt.close()
            
            st.write("""
            **How to read this plot:**
            - Base value: Average house price the model learned
            - Red bars push the prediction higher
            - Blue bars push the prediction lower
            - Final prediction shown at the top
            """)
            
            # Bar plot for feature importance
            st.write("#### 📊 Feature Importance (Overall Impact)")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                st.session_state.prepared_data,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig2, bbox_inches='tight')
            plt.close()
            
            st.info("""
            **What is SHAP?**
            
            SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of machine learning models.
            It shows how each feature contributed to the final prediction.
            
            This makes the model's decisions transparent and interpretable — critical for real-world ML applications!
            """)
            
        except Exception as e:
            st.error(f"🚨 SHAP Error: {e}")
            st.info("💡 This might take a few seconds on first run while SHAP initializes.")
