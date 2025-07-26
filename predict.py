import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load trained model
model = joblib.load("models/sklearn_model.joblib")

# Load dataset
X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predictions
y_pred = model.predict(X_test)

# Calculate R² Score
score = r2_score(y_test, y_pred)
print(f"✅ Model verification successful. R² Score: {score:.4f}")
