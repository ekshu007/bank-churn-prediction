import streamlit as st
import requests

# Set page configuration
st.set_page_config(page_title="Bank Customer Churn Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Enter the customer's profile details below to predict their likelihood of leaving the bank.")

# Create a two-column layout for the input form
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=40)
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=60000.0, step=1000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)

with col2:
    country = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    products_number = st.selectbox("Number of Bank Products", [1, 2, 3, 4])
    credit_card = st.radio("Has Credit Card?", ["Yes", "No"])
    active_member = st.radio("Is Active Member?", ["Yes", "No"])

st.markdown("---")

# Prediction Button
if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    # Map Yes/No to 1/0
    cc_val = 1 if credit_card == "Yes" else 0
    active_val = 1 if active_member == "Yes" else 0
    
    # Construct payload for our FastAPI backend
    payload = {
        "credit_score": credit_score,
        "country": country,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": cc_val,
        "active_member": active_val,
        "estimated_salary": estimated_salary
    }
    
    try:
        # Call the FastAPI endpoint
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prob = result["churn_probability"] * 100
            risk = result["risk_level"]
            
            # Display results with dynamic styling
            if risk == "High Risk":
                st.error(f"⚠️️ **{risk}**")
                st.write(f"This customer has a **{prob:.1f}%** probability of churning.")
                st.progress(result["churn_probability"])
            else:
                st.success(f"✅ **{risk}**")
                st.write(f"This customer has a **{prob:.1f}%** probability of churning.")
                st.progress(result["churn_probability"])
        else:
            st.warning("Error getting prediction from backend.")
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure FastAPI is running on port 8000.")