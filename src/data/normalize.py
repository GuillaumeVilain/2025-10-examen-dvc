import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

input_train_path = "data/processed_data/X_train.csv"
input_test_path = "data/processed_data/X_test.csv"
output_dir = "data/processed_data"

# Chargement
X_train = pd.read_csv(input_train_path)
X_test = pd.read_csv(input_test_path)

# Séparer les colonnes numériques et non numériques
numeric_cols = X_train.select_dtypes(include=["number"]).columns
non_numeric_cols = X_train.select_dtypes(exclude=["number"]).columns

# Scaler
scaler = StandardScaler()

# Ajust/transform
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# DataFrames (conversion)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

os.makedirs(output_dir, exist_ok=True)
X_train_scaled.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
X_test_scaled.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)

print("Normalisation : done")
print(f"Fichiers sauvegardés dans {output_dir}/")
