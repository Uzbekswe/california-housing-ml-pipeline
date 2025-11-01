from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib, pandas as pd
from pathlib import Path
import sys

# Import custom transformers
from serve.transformers import CombinedAttributesAdder, DataFrameSelector

# Monkey-patch: Add transformers to __main__ so joblib can find them
import __main__
__main__.CombinedAttributesAdder = CombinedAttributesAdder
__main__.DataFrameSelector = DataFrameSelector

# ==== Input schema ====
class HousingInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str = Field(examples=["INLAND","NEAR BAY","<1H OCEAN","NEAR OCEAN","ISLAND"])

# ==== App ====
app = FastAPI(title="California Housing Price Prediction API")

# Global variable for pipeline
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    model_path = Path(__file__).resolve().parents[1] / "artifacts" / "california_price_pipeline.joblib"
    pipeline = joblib.load(model_path)

@app.get("/")
def index():
    return {"status": "running", "model": "california_housing"}

@app.post("/predict")
def predict(data: HousingInput):
    df = pd.DataFrame([data.model_dump()])
    prediction = pipeline.predict(df)[0]
    return {"prediction": float(prediction)}
