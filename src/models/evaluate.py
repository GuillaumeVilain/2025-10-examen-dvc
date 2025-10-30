import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


with open("models/en_model.pkl", "rb") as f:
    model = pickle.load(f)

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
X_test = X_test.drop(columns=["date"], errors="ignore")

y_test = pd.read_csv("data/processed_data/y_test.csv").iloc[:, 0]

# Predict
y_pred = model.predict(X_test)

# Save predictions
os.makedirs(os.path.dirname("data/data_predictions.csv"), exist_ok=True)
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("data/data_predictions.csv", index=False)

# --- Compute metrics ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

metrics = {
    "mse": mse,
    "rmse": rmse,
    "r2": r2
}

# --- Save metrics ---
os.makedirs(os.path.dirname("metrics/scores.json"), exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)

# --- Optional print for log visibility ---
print(json.dumps(metrics, indent=4))