import os, joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Train model
data = fetch_california_housing()
X, y = data.data, data.target
model = LinearRegression().fit(X, y)

# Save model
joblib.dump(model, "models/sklearn_model.joblib")
print("✅ Model saved at models/sklearn_model.joblib")
