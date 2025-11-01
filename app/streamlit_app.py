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

# === Conversion Presets for User-Friendly Inputs ===
CONVERSION_PRESETS = {
    "house_size": {
        "Small (1-2 BR)": {"rooms": 5, "bedrooms": 2},
        "Medium (3 BR)": {"rooms": 7, "bedrooms": 3},
        "Large (4+ BR)": {"rooms": 10, "bedrooms": 4}
    },
    "area_type": {
        "Urban (City)": {"households": 200, "density": 3.2},
        "Suburban": {"households": 120, "density": 2.5},
        "Rural": {"households": 50, "density": 2.8}
    },
    "income_level": {
        "Low ($30-50k)": 4.0,
        "Middle ($50-80k)": 6.5,
        "High ($80k+)": 9.0
    },
    "house_age": {
        "New (0-10 yrs)": 5,
        "Modern (10-30 yrs)": 20,
        "Old (30+ yrs)": 45
    }
}

def convert_simple_to_census(house_size, area_type, income_level, house_age, ocean_proximity, latitude, longitude):
    """Convert user-friendly inputs to census block data"""
    # Get presets
    size_preset = CONVERSION_PRESETS["house_size"][house_size]
    area_preset = CONVERSION_PRESETS["area_type"][area_type]
    income = CONVERSION_PRESETS["income_level"][income_level]
    age = CONVERSION_PRESETS["house_age"][house_age]
    
    # Calculate census block aggregates
    households = area_preset["households"]
    total_rooms = size_preset["rooms"] * households
    total_bedrooms = size_preset["bedrooms"] * households
    population = area_preset["density"] * households
    
    return {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": income,
        "ocean_proximity": ocean_proximity
    }

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
if 'census_data' not in st.session_state:
    st.session_state.census_data = None

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
tab1, tab2, tab3, tab4 = st.tabs(["🏡 Predict Price", "🌍 Map View", "🔍 Explain Prediction", "📁 Batch Upload"])

# === TAB 1: Prediction ===
with tab1:
    # Input mode selector
    input_mode = st.radio(
        "**Choose Input Mode:**",
        ["🏡 Simple Mode (Recommended)", "📊 Advanced Mode (Census Data)"],
        horizontal=True,
        help="Simple mode for individual houses, Advanced mode for census block data"
    )
    
    if input_mode == "🏡 Simple Mode (Recommended)":
        st.write("### Enter Your Property Details")
        st.write("*We'll estimate neighborhood statistics based on your inputs*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📍 Location**")
            latitude = st.number_input("Latitude", value=37.88, key="lat_simple", 
                                      help="North-South position (32-42 for California)")
            longitude = st.number_input("Longitude", value=-122.23, key="long_simple",
                                       help="East-West position (-125 to -114 for California)")
            ocean_proximity = st.selectbox(
                "Ocean Proximity",
                ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
                index=0,
                help="How close to the ocean?"
            )
            
            st.write("**🏠 Your House**")
            house_size = st.selectbox(
                "House Size",
                list(CONVERSION_PRESETS["house_size"].keys()),
                index=1,
                help="Approximate size based on bedrooms"
            )
            house_age = st.selectbox(
                "House Age",
                list(CONVERSION_PRESETS["house_age"].keys()),
                index=1,
                help="How old is the property?"
            )
        
        with col2:
            st.write("**🌆 Neighborhood**")
            area_type = st.selectbox(
                "Area Type",
                list(CONVERSION_PRESETS["area_type"].keys()),
                index=1,
                help="Urban = city center, Suburban = residential, Rural = countryside"
            )
            income_level = st.selectbox(
                "Neighborhood Income Level",
                list(CONVERSION_PRESETS["income_level"].keys()),
                index=1,
                help="Average household income in your area"
            )
            
            # Show what values are being used
            with st.expander("🔍 See how we estimate your neighborhood"):
                size_preset = CONVERSION_PRESETS["house_size"][house_size]
                area_preset = CONVERSION_PRESETS["area_type"][area_type]
                income_val = CONVERSION_PRESETS["income_level"][income_level]
                age_val = CONVERSION_PRESETS["house_age"][house_age]
                
                st.write(f"""
                **Your house:**
                - Bedrooms: {size_preset['bedrooms']}
                - Approximate rooms: {size_preset['rooms']}
                - Age: {age_val} years (median for neighborhood)
                
                **Neighborhood estimates:**
                - Type: {area_type}
                - Similar homes in block: ~{area_preset['households']}
                - People per household: ~{area_preset['density']:.1f}
                - Median income: ${income_val * 10000:,.0f}/year
                
                **Census block aggregates (what the model sees):**
                - Total rooms (all houses): {size_preset['rooms'] * area_preset['households']:,.0f}
                - Total bedrooms (all houses): {size_preset['bedrooms'] * area_preset['households']:,.0f}
                - Population: {int(area_preset['density'] * area_preset['households'])}
                - Households: {area_preset['households']}
                """)
        
        # Convert to census data
        census_data = convert_simple_to_census(
            house_size, area_type, income_level, house_age, 
            ocean_proximity, latitude, longitude
        )
        
    else:  # Advanced mode
        st.write("### Enter Census Block Data")
        st.info("💡 These are aggregate statistics for an entire neighborhood/census block, not a single house")
        
        col1, col2 = st.columns(2)

        with col1:
            longitude = st.number_input("Longitude", value=-122.23, key="long_adv",
                                       help="East-West coordinate")
            latitude = st.number_input("Latitude", value=37.88, key="lat_adv",
                                      help="North-South coordinate")
            housing_median_age = st.number_input("Housing Median Age (years)", value=41.0,
                                                help="Median age of all houses in the block")
            total_rooms = st.number_input("Total Rooms (entire block)", value=880.0,
                                         help="Sum of ALL rooms across ALL houses in the block")

        with col2:
            total_bedrooms = st.number_input("Total Bedrooms (entire block)", value=129.0,
                                            help="Sum of ALL bedrooms across ALL houses")
            population = st.number_input("Population (entire block)", value=322.0,
                                        help="Total number of residents in the census block")
            households = st.number_input("Households (number of homes)", value=126.0,
                                        help="Number of separate homes/families in the block")
            median_income = st.number_input("Median Income (÷ $10k)", value=8.3252,
                                           help="Median household income divided by $10,000 (e.g., 8.3 = $83,000)")
            ocean_proximity = st.selectbox(
                "Ocean Proximity",
                ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
                index=0,
                help="Distance to ocean/bay"
            )
        
        # Use advanced inputs directly
        census_data = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }

    # === Predict using loaded model ===
    if st.button("🚀 Predict Price", type="primary"):
        # Prepare input data from census_data dictionary
        input_data = pd.DataFrame([census_data])

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
            st.session_state.census_data = census_data  # Store for PDF generation
            
            st.success(f"### 💰 Estimated Median House Value: **${prediction:,.2f}**")

        except Exception as e:
            st.error(f"🚨 Prediction Error: {e}")
            st.info("💡 Make sure the model file exists in the artifacts/ directory")

    # === PDF Download Button (appears after any prediction) ===
    if st.session_state.prediction is not None and st.session_state.census_data is not None:
        st.write("---")  # Divider line
        
        if st.button("📄 Download Prediction Report as PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                c = canvas.Canvas(tmp.name, pagesize=letter)

                c.setFont("Helvetica-Bold", 16)
                c.drawString(30, 750, "California Housing Price Prediction Report")
                
                c.setFont("Helvetica", 10)
                c.drawString(30, 730, "Generated by ML Pipeline")

                c.setFont("Helvetica", 12)
                y = 690
                
                cd = st.session_state.census_data
                fields = {
                    "Longitude": cd["longitude"],
                    "Latitude": cd["latitude"],
                    "Housing Median Age": cd["housing_median_age"],
                    "Total Rooms": cd["total_rooms"],
                    "Total Bedrooms": cd["total_bedrooms"],
                    "Population": cd["population"],
                    "Households": cd["households"],
                    "Median Income": f"${cd['median_income'] * 10000:,.0f}",
                    "Ocean Proximity": cd["ocean_proximity"],
                    "Predicted House Value": f"${st.session_state.prediction:,.2f}",
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

    # === Neighborhood Stats Section ===
    if st.session_state.prediction and st.session_state.census_data:
        st.write("### 📊 Neighborhood Facts")
        
        cd = st.session_state.census_data
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Median Income",
                value=f"${cd['median_income'] * 10000:,.0f}",
                delta="Regional data"
            )
        
        with col2:
            st.metric(
                label="Population Density",
                value=f"{cd['population']/cd['households']:.1f}",
                delta="People per household"
            )
        
        with col3:
            st.metric(
                label="Avg Rooms",
                value=f"{cd['total_rooms']/cd['households']:.1f}",
                delta="Per household"
            )
        
        st.info("""
        📍 **This prediction is based on:**  
        • Neighborhood aggregate data (census block level)
        • Similar homes in the area
        • Local market conditions
        """)
    else:
        st.info("💡 Enter property details above and click 'Predict Price' to see results")

# === TAB 2: Map View ===
with tab2:
    st.write("### 🗺 Property Location")
    
    # Map style selector
    map_style = st.selectbox("Map Style", ["Street Map", "Satellite"])
    
    # Create map centered on the property location
    m = folium.Map(
        location=[latitude, longitude], 
        zoom_start=12
    )
    
    # Add satellite tiles if selected
    if map_style == "Satellite":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite"
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles="OpenStreetMap",
            name="Street Map"
        ).add_to(m)
    
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

# === TAB 4: Batch Upload ===
with tab4:
    st.write("### 📁 Upload CSV for Batch Predictions")
    
    st.info("""
    **CSV Format Required:**
    Your CSV must include these columns:
    - `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`
    - `population`, `households`, `median_income`, `ocean_proximity`
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### 📋 Preview of Uploaded Data:")
            st.dataframe(df.head(10))
            
            st.write(f"**Total rows:** {len(df)}")
            
            if st.button("🚀 Run Batch Predictions", type="primary"):
                with st.spinner("Running predictions..."):
                    # Load pipeline and predict
                    pipeline = load_model()
                    predictions = pipeline.predict(df)
                    df["predicted_house_value"] = predictions
                    
                    st.success(f"✅ Predictions completed for {len(df)} properties!")
                    
                    # Show results preview
                    st.write("### 📊 Results Preview:")
                    st.dataframe(df.head(10))
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Predicted Price", f"${predictions.mean():,.2f}")
                    with col2:
                        st.metric("Minimum Price", f"${predictions.min():,.2f}")
                    with col3:
                        st.metric("Maximum Price", f"${predictions.max():,.2f}")
                    
                    # Download button
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv_data,
                        file_name="housing_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"⚠️ Error processing CSV: {e}")
            st.info("💡 Make sure your CSV has all required columns with correct names.")
