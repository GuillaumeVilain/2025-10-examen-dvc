import pickle
import pandas as pd
from sklearn.linear_model import ElasticNet

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
X_train = X_train.drop(columns=["date"])
y_train = pd.read_csv("data/processed_data/y_train.csv")

# y -> 1D
if y_train.shape[1] > 1:
    y_train = y_train.iloc[:, 0]
else:
    y_train = y_train.squeeze()

with open("models/best_params.pkl", "rb") as f:
    payload = pickle.load(f)

best_params = payload["best_params"]

# Fit ElasticNet
model = ElasticNet(**best_params)
model.fit(X_train, y_train)

with open("models/en_model.pkl", "wb") as f:
    pickle.dump(model, f)