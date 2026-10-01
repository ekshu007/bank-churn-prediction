from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import uvicorn

app = FastAPI(title="Bank Churn Prediction API")

# Dynamically set paths so it runs regardless of where the terminal is opened
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_churn_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "standard_scaler.pkl"

# Load the saved artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define the expected JSON payload
class CustomerData(BaseModel):
    credit_score: int
    country: str
    gender: str
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float
@app.get("/")
def read_root():
    return {"message": "Bank Churn Prediction API is running. Visit /docs to test endpoints."}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    # 1. Manually One-Hot Encode the categorical variables just like pd.get_dummies did
    country_Germany = 1 if customer.country.lower() == 'germany' else 0
    country_Spain = 1 if customer.country.lower() == 'spain' else 0
    gender_Male = 1 if customer.gender.lower() == 'male' else 0

    # 2. Construct the dataframe in the exact column order the model was trained on
    input_data = pd.DataFrame([{
        'credit_score': customer.credit_score,
        'age': customer.age,
        'tenure': customer.tenure,
        'balance': customer.balance,
        'products_number': customer.products_number,
        'credit_card': customer.credit_card,
        'active_member': customer.active_member,
        'estimated_salary': customer.estimated_salary,
        'country_Germany': country_Germany,
        'country_Spain': country_Spain,
        'gender_Male': gender_Male
    }])

    # 3. Scale the numerical features using our saved StandardScaler
    input_scaled = scaler.transform(input_data)

    # 4. Generate the prediction and probability
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(model.predict(input_scaled)[0])

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction,
        "risk_level": "High Risk" if probability > 0.5 else "Low Risk"
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)