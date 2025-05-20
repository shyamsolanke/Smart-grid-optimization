import pandas as pd
import numpy as np
import cupy as cp
from solver_gpu import solve_theta_gpu
from preprocessing import build_lagged_matrix_gpu
import pulp  # PuLP for MILP
import math

# 1) Setup
df = pd.read_csv('smart_grid_dataset.csv')
y_np = df['Predicted Load (kW)'].values
exog_cols = ['Voltage (V)', 'Current (A)', 'Power Factor',
    'Solar Power (kW)', 'Wind Power (kW)', 'Grid Supply (kW)', 'Temperature (°C)', 'Humidity (%)', 'Electricity Price (USD/kWh)']   # same nine columns
exog_np   = df[exog_cols].values

p_values = list(range(50, 301, 10))
b_values = list(range(50, 301, 10))
I, J = len(p_values), len(b_values)

# 2) Run your original GPU grid‐search and _collect_ results
costs     = np.zeros((I,J))
thetas    = [[None]*J for _ in range(I)]
phis      = [[None]*J for _ in range(I)]
etas      = [[None]*J for _ in range(I)]

for i, p in enumerate(p_values):
    for j, b in enumerate(b_values):
        A_np, y_t_np = build_lagged_matrix_gpu(y_np, exog_np, p, b)
        # … train/test split, to GPU, solve … exactly your code …
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
        theta_gpu = solve_theta_gpu(A_train, y_train)
        # theta = cp.asnumpy(theta_gpu)
        # compute rmse exactly as you do
        residuals = A_test.dot(theta_gpu) - y_test
        rmse = math.sqrt(float(cp.mean(residuals**2).item()))
        costs[i,j] = rmse
        thetas[i][j] = theta_gpu
        phis[i][j]   = theta_gpu[:p]
        etas[i][j]   = theta_gpu[p:].reshape(b, exog_np.shape[1])

# build problem
prob = pulp.LpProblem("select_best_lag", pulp.LpMinimize)
x = pulp.LpVariable.dicts("x", (range(I), range(J)), cat="Binary")

# objective
prob += pulp.lpSum(costs[i][j] * x[i][j] for i in range(I) for j in range(J))

# must pick exactly one (i,j)
prob += pulp.lpSum(x[i][j] for i in range(I) for j in range(J)) == 1

# big-M constraints on φ and η (box constraints)
for i in range(I):
    for j in range(J):
        if thetas[i][j] is None:
            continue
        # ensure φ entries are plain floats
        phi_vals = phis[i][j]
        if isinstance(phi_vals, cp.ndarray):
            phi_vals = cp.asnumpy(phi_vals)
        # similarly for eta
        eta_vals = etas[i][j]
        if isinstance(eta_vals, cp.ndarray):
            eta_vals = cp.asnumpy(eta_vals)

        # now phi_vals and eta_vals are NumPy arrays → iterate
        for φ in phi_vals.flatten():
            prob += float(φ) * x[i][j] <=  1
            prob += -float(φ) * x[i][j] <=  1

        for η in eta_vals.flatten():
            prob += float(η) * x[i][j] <=  1
            prob += -float(η) * x[i][j] <=  1

# solve…
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# 5) Extract the chosen (p*,b*) and θ
for i in range(I):
    for j in range(J):
        if pulp.value(x[i][j]) > 0.5:
            p_star = p_values[i]
            b_star = b_values[j]
            theta_star = thetas[i][j]

print(f"Selected p*={p_star}, b*={b_star}, RMSE={costs[i,j]:.4f}")
