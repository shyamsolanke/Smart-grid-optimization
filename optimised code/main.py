import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from solver_gpu import solve_theta_gpu
import joblib
from preprocessing import build_lagged_matrix_gpu
import json
import math

# 1) Load data with pandas (NumPy arrays)
df = pd.read_csv('smart_grid_dataset.csv')
y_np = df['Predicted Load (kW)'].values
exog_cols = [
    'Voltage (V)', 'Current (A)', 'Power Factor',
    'Solar Power (kW)', 'Wind Power (kW)', 'Grid Supply (kW)',
    'Temperature (°C)', 'Humidity (%)', 'Electricity Price (USD/kWh)'
]
exog_np = df[exog_cols].values

# 2) Search settings
p_values = list(range(50, 301, 50))   # [50,100,150,200,250,300]
b_values = list(range(50, 301, 50))   # same

# 2) Preallocate a costs grid of the right shape
costs = np.zeros((len(p_values), len(b_values)))

best = {'cost': np.inf, 'p': None, 'b': None, 'theta': None}

# 3) Loop over those lists
for i, p in enumerate(p_values):
    for j, b in enumerate(b_values):
        # Build design matrix in NumPy
        A_np, y_t_np = build_lagged_matrix_gpu(y_np, exog_np, p, b)
        
        # Train/test split indices (NumPy)
        N = len(y_t_np)
        idx = np.random.RandomState(42).permutation(N)
        split = int(0.8 * N)
        train_idx, test_idx = idx[:split], idx[split:]
        
        # 1) Convert train/test splits to CuPy arrays
        A_train = cp.asarray(A_np[train_idx], dtype=cp.float64)
        y_train = cp.asarray(y_t_np[train_idx], dtype=cp.float64)
        A_test  = cp.asarray(A_np[test_idx],  dtype=cp.float64)
        y_test  = cp.asarray(y_t_np[test_idx],  dtype=cp.float64)

        # 2) Call your helper which internally does (A^T A) and (A^T y)
        theta_gpu = solve_theta_gpu(A_train, y_train)
        
        # GPU: compute test residuals and cost
        residuals = A_test.dot(theta_gpu) - y_test
        mse_test = float(cp.mean(residuals**2).item())    # mean squared error
        rmse_test = math.sqrt(mse_test)                   # root mean squared error
        cost_test = rmse_test
        
        # Record cost in NumPy grid
        costs[i, j] = cost_test
        
        # Update best if improved
        if cost_test < best['cost']:
            # Move theta back to CPU
            best.update({
                'cost': cost_test,
                'p': p,
                'b': b,
                'theta': cp.asnumpy(theta_gpu)
            })

# 4) Unpack best model
p_star = best['p']
b_star = best['b']
theta_star = best['theta']
phi_star = theta_star[:p_star]
eta_star = theta_star[p_star:].reshape(b_star, exog_np.shape[1])

# After you’ve found p_star, b_star, phi_star, eta_star:
best_model = {
    'p':     p_star,
    'b':     b_star,
    'phi':   phi_star,   # 1D array of length p_star
    'eta':   eta_star    # 2D array of shape (b_star, n_features)
}

best_model_json = {
    'p':     p_star,
    'b':     b_star,
    'phi':   phi_star.tolist(),   # 1D array of length p_star
    'eta':   eta_star.tolist()   # 2D array of shape (b_star, n_features)
}

# Save the best model to a JSON file
with open('best_arx_model.json', 'w') as f:
    json.dump(best_model_json, f)

# Write to disk
joblib.dump(best_model, 'best_arx_model.pkl')
print("Saved best model to best_arx_model.pkl")

print(f"Best lag orders → p = {p_star}, b = {b_star}")
print(f"Minimum Test SSE = {best['cost']:.3f}")
print("φ coefficients:", phi_star)
print("η coefficients (b × features):\n", eta_star)

# 5) Plot heatmap (NumPy)
plt.figure()
plt.imshow(costs, origin='lower', aspect='auto')
plt.colorbar(label='Test RMSE')
plt.xticks(range(len(b_values)), b_values)
plt.yticks(range(len(p_values)), p_values)
plt.xlabel('b (exog lags)')
plt.ylabel('p (AR lags)')
plt.title('Test RMSE across lag orders p and b (GPU-accelerated)')
plt.show()
