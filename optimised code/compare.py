# compare_with_sarimax.py

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import cupy as cp

from preprocessing import build_lagged_matrix_gpu
from solver_gpu    import solve_theta_gpu  # your GPU ARX solver

def main():
    # 1) Load best lag orders from JSON
    with open("best_arx_model.json","r") as f:
        jm = json.load(f)
    p_star = jm["p"]
    b_star = jm["b"]
    print(f"Using p*={p_star}, b*={b_star}")

    # 2) Load raw data
    df = pd.read_csv("smart_grid_dataset.csv")
    y_np     = df["Predicted Load (kW)"].values
    exog_cols = [
        "Voltage (V)", "Current (A)", "Power Factor",
        "Solar Power (kW)", "Wind Power (kW)", "Grid Supply (kW)",
        "Temperature (°C)", "Humidity (%)", "Electricity Price (USD/kWh)"
    ]
    exog_np  = df[exog_cols].values

    # 3) Build A, y_target on GPU
    A_gpu, y_gpu = build_lagged_matrix_gpu(y_np, exog_np, p_star, b_star)

    # 4) Slice first 5000 for training, next 200 for test
    N_train = 5000
    N_test  =  200

    A_train_gpu = A_gpu[:N_train]
    y_train_gpu = y_gpu[:N_train]

    A_test_gpu  = A_gpu[N_train:N_train+N_test]
    y_test_gpu  = y_gpu[N_train:N_train+N_test]

    # 5) Retrain ARX on GPU
    print("Retraining ARX on 5000 samples (GPU)...")
    theta_gpu = solve_theta_gpu(A_train_gpu, y_train_gpu)

    # 6) Predict on test set (GPU → CPU)
    y_pred_custom_gpu = A_test_gpu.dot(theta_gpu)
    y_pred_custom     = cp.asnumpy(y_pred_custom_gpu)
    y_test            = cp.asnumpy(y_test_gpu)
    rmse_custom       = np.sqrt( np.mean((y_test - y_pred_custom)**2) )
    print(f"Custom ARX RMSE: {rmse_custom:.4f}")

    # 7) Prepare NumPy arrays for SARIMAX
    A_train = cp.asnumpy(A_train_gpu)
    y_train = cp.asnumpy(y_train_gpu)
    A_test  = cp.asnumpy(A_test_gpu)
    y_test  = cp.asnumpy(y_test_gpu)

    # exogenous slices drop first p_star columns
    exog_train = A_train[:, p_star:]
    exog_test  = A_test[:,  p_star:]

    # 8) Fit SARIMAX on the same 5000 train points
    print("Fitting SARIMAX on 5000 samples (CPU)...")
    sarimax = SARIMAX(
        endog=y_train,
        exog=exog_train,
        order=(p_star,0,0)
        # enforce_stationary=False,
        # enforce_invertible=False
    )
    res = sarimax.fit(disp=False)

    # 9) Forecast and compute RMSE
    y_pred_sarimax = res.predict(start=0, end=N_test-1, exog=exog_test)
    rmse_sarimax   = np.sqrt( np.mean((y_test - y_pred_sarimax)**2) )
    print(f"SARIMAX    RMSE: {rmse_sarimax:.4f}")

    # 10) Plot first 200 test points
    plt.figure(figsize=(10,5))
    plt.plot(y_test,           label="Actual",       linewidth=2)
    plt.plot(y_pred_custom,    label="Custom ARX",    linestyle="--")
    plt.plot(y_pred_sarimax,   label="SARIMAX",       linestyle=":")
    plt.title("Load Forecast on 200-sample Test Set")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
