import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load features
features = pickle.load(open("features.pkl", "rb"))

# Example customer
input_dict = {
    "tenure": 72,
    "MonthlyCharges": 20,
    "Contract_Month-to-month": 0,
    "Contract_One year": 0,
    "Contract_Two year": 1
}

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

if prediction == 1:
    print("Customer likely to churn")
else:
    print("Customer likely to stay")

print(f"Churn Probability: {probability:.2f}")