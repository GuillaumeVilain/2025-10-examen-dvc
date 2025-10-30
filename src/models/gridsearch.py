import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
import pickle
import numpy as np

X = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_df = pd.read_csv("data/processed_data/y_train.csv")
y = y_df.iloc[:, 0]

# Check ndarray float
X = X.drop(columns=["date"])
X = X.to_numpy(dtype=float)
y = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

model = ElasticNet(max_iter=5000)

param_grid = {
    "alpha": np.logspace(-4, 2, 20),      # régularisation L1+L2
    "l1_ratio": np.linspace(0.05, 0.95, 19),  # mélange L1/L2 
    "fit_intercept": [True, False]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",  # minimise la RMSE
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=0,
    return_train_score=False,
)

# --- Entraînement ---
search.fit(X, y)

best_params = search.best_params_
best_score = -search.best_score_  # RMSE positive

# --- Sauvegarde des meilleurs paramètres ---
payload = {
    "model": "ElasticNet",
    "scoring": "RMSE",
    "best_params": best_params,
    "best_cv_rmse": best_score,
    "n_splits": cv.get_n_splits(),
}

with open("models/best_params.pkl", "wb") as f:
    pickle.dump(payload, f)

# Print console
print(f"Meilleurs paramètres : {best_params}")
print(f"CV RMSE (moyenne des folds) : {best_score:.6f}")
print(f"Fichier sauvegardé : models/best_params.pkl")