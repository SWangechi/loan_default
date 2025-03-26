import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Define request data format
class LoanApplication(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int

# Load the trained model and scaler
model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define expected feature names
FEATURE_NAMES = ["age", "income", "loan_amount", "credit_score"]

# Initialize FastAPI app
app = FastAPI(title="Loan Default Prediction API", description="Predicts loan default probability", version="1.0")

# Define home route
@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running!"}

# Define prediction route
@app.post("/predict/")
async def predict_default(application: LoanApplication):
    try:
        # Convert input to DataFrame with column names
        input_data = pd.DataFrame([[application.age, application.income, application.loan_amount, application.credit_score]], 
                                  columns=FEATURE_NAMES)

        # Scale the input
        scaled_data = scaler.transform(input_data)

        # Predict probability
        probability = model.predict_proba(scaled_data)[0][1]

        return {"loan_default_probability": round(float(probability), 2)}

    except Exception as e:
        return {"error": str(e)}
        
# Run the API with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
