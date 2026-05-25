import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import load_and_preprocess_data

# Load dataset
df = load_and_preprocess_data(
    "../data/churn.csv"
)

# Features
X = df.drop("Churn", axis=1)

# Target
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train better model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print(f"Accuracy: {acc:.2f}")

# Save model
pickle.dump(
    model,
    open("../model.pkl", "wb")
)

# Save features
pickle.dump(
    X.columns.tolist(),
    open("../features.pkl", "wb")
)

print("Model saved successfully!")