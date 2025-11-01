import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformers (required for loading the pipeline)
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self # nothing to do here
    
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
        return self # do nothing
    
    def transform(self, X, y=None):
        return X[self.attibute_names].values

# Load model
model_path = Path(__file__).resolve().parents[1] / "california_price_pipeline.joblib"
pipeline = joblib.load(model_path)

# Example input:
# python predict_cli.py '{"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41,
# "total_rooms": 880, "total_bedrooms": 129, "population": 322,
# "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"}'

raw = json.loads(sys.argv[1])
df = pd.DataFrame([raw])

pred = pipeline.predict(df)[0]
print(f"Predicted value: {pred:.2f}")
