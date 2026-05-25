import streamlit as st
import pickle
import pandas as pd

# Title
st.title("Telecom Churn Predictor")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load feature names
features = pickle.load(open("features.pkl", "rb"))

# User inputs
tenure = st.slider(
    "Customer Tenure",
    0,
    72,
    12
)

monthly_charges = st.slider(
    "Monthly Charges",
    0.0,
    150.0,
    70.0
)

contract = st.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

# Prediction button
if st.button("Predict Churn"):

    # One-hot encoding
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract_Month-to-month": 0,
        "Contract_One year": 0,
        "Contract_Two year": 0
    }

    # Set selected contract
    if contract == "Month-to-month":
        input_dict["Contract_Month-to-month"] = 1

    elif contract == "One year":
        input_dict["Contract_One year"] = 1

    else:
        input_dict["Contract_Two year"] = 1

    # Convert to DataFrame
    input_data = pd.DataFrame([input_dict])

    # Match columns
    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    # Show result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")

    st.write(
        f"Churn Probability: {probability:.2f}"
    )