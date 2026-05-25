import pandas as pd

def load_and_preprocess_data(filepath):

    # Load dataset
    df = pd.read_csv(filepath)

    # Keep useful columns
    df = df[[
        "tenure",
        "MonthlyCharges",
        "Contract",
        "Churn"
    ]]

    # Ensure MonthlyCharges is numeric
    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce"
    )

    # Drop missing values
    df.dropna(inplace=True)

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=["Contract"]
    )

    # Convert target
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    return df