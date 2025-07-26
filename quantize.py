import joblib
import numpy as np
import os

# Load sklearn model
model = joblib.load("models/sklearn_model.joblib")
coef = model.coef_
intercept = model.intercept_

# Save unquantized
unquant = {"coef": coef, "intercept": intercept}
joblib.dump(unquant, "models/unquant_params.joblib")

# Quantization
scale = 255 / (np.max(np.abs(coef)) + 1e-8)
q_coef = (coef * scale).astype(np.uint8)
q_intercept = int(intercept * scale)

quant = {"coef": q_coef, "intercept": q_intercept, "scale": scale}
joblib.dump(quant, "models/quant_params.joblib")

# Dequantization & Inference
X, y = joblib.load("models/test_data.joblib") if os.path.exists("models/test_data.joblib") else (None, None)
if X is None:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    X, y = fetch_california_housing(return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)
    joblib.dump((X_test, y_test), "models/test_data.joblib")

# Manual inference
recon_coef = q_coef.astype(np.float32) / scale
recon_intercept = q_intercept / scale
y_pred = X_test @ recon_coef + recon_intercept

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(f"✅ R² Score (quantized model): {score:.4f}")
