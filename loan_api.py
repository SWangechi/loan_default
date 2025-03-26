import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model and scaler
model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI app
app = FastAPI(title="Loan Default Prediction API", description="Predicts loan default probability", version="1.0")

# Define request data format
class LoanInput(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int

# Define home route
@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running!"}

# Define prediction route
@app.post("/predict/")
def predict_default(data: LoanInput):
    try:
        # Convert input data to a NumPy array
        input_data = np.array([[data.age, data.income, data.loan_amount, data.credit_score]])

        # Standardize input data using the saved scaler
        input_scaled = scaler.transform(input_data)

        # Predict the probability of loan default
        probability = model.predict_proba(input_scaled)[:, 1][0]

        return {"loan_default_probability": round(float(probability), 4)}
    
    except Exception as e:
        return {"error": str(e)}

# Run the API with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
