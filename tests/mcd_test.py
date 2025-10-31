# test_mcd_tf.py
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
# Your package
from uq_calculator import get_mcd, get_ece, get_nll

# --------------------
# 1) Paths & data load
# --------------------
DATA_PATH = "/Users/kayleesmith/uq_calculator/tests/filtered copy/models"
MODEL_NAME = "LSTM MCD1"
MODEL_PATH = ("/Users/kayleesmith/uq_calculator/tests/filtered copy/models/LSTM_MCD1.keras")

X_val = np.load("/Users/kayleesmith/uq_calculator/tests/filtered copy/X_validate.npy")
loc_val = np.load("/Users/kayleesmith/uq_calculator/tests/filtered copy/location_validate.npy")
mon_val = np.load("/Users/kayleesmith/uq_calculator/tests/filtered copy/month_validate.npy")
day_val = np.load("/Users/kayleesmith/uq_calculator/tests/filtered copy/day_validate.npy")
y_val  = np.load("/Users/kayleesmith/uq_calculator/tests/filtered copy/y_validate.npy")

print(f" Data loaded:"
      f"\n  X_validate: {X_val.shape}"
      f"\n  location_validate: {loc_val.shape}"
      f"\n  month_validate: {mon_val.shape}"
      f"\n  day_validate: {day_val.shape}"
      f"\n  y_validate: {y_val.shape}")


# 2) Load the model
print(f"\nLoading model from: {MODEL_PATH}")
# compile=False so we don't need custom metric objects at load time
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.")


# 3) Run MC Dropout

inputs = [X_val, loc_val, mon_val, day_val]

print("\n Running Monte Carlo Dropout sampling (n=100)...")
mean_pred, var_pred, all_preds = get_mcd(
    model,
    x_data=inputs,
    n_samples=1,
    framework="tf"
)
print("MC sampling complete.")


# 4) Metrics
# Shapes typically: (N, 24, 1). Flatten to compare.
y_true_flat  = y_val.reshape(-1)
mean_flat    = mean_pred.reshape(-1)
std_flat     = np.sqrt(var_pred.reshape(-1) + 1e-12)  # numerical safety

rmse = np.sqrt(mean_squared_error(y_true_flat, mean_flat))
r2   = r2_score(y_true_flat, mean_flat)
nll = get_nll(y_true_flat, mean_flat, std_flat)
ece  = get_ece(y_true_flat, mean_flat, std_flat)

print("\n📊 Validation metrics")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²  : {r2:.4f}")
print(f"  NLL : {nll:.4f}")
print(f"  ECE : {ece:.4f}")

