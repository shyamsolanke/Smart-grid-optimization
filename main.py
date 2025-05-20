import pandas as pd
import preprocessing as pp
import solver
import numpy as np

# 1) Load your data
df = pd.read_csv('smart_grid_dataset.csv')
y = df['Predicted Load (kW)'].values
exog_cols = ['Voltage (V)', 'Current (A)','Power Factor',
             'Solar Power (kW)','Wind Power (kW)', 'Grid Supply (kW)',
             'Temperature (°C)','Humidity (%)','Electricity Price (USD/kWh)']
exog = df[exog_cols].values

# 2) Fix p and b here:
p = 4   # e.g. use 4 lags of X_t
b = 6   #    and 6 lags of each exogenous feature


A, y_target = pp.build_lagged_matrix(y, exog, p, b)

theta = solver.solve_theta(A, y_target)
phi = theta[:p]
eta = theta[p:]

print("phi coefficients:", phi)
print("eta  coefficients:", eta)




# # # 4) Solve for φ and η in closed form
# # theta, *_ = np.linalg.lstsq(A, y_target, rcond=None)
# phi = theta[:p]
# eta = theta[p:].reshape(b, exog.shape[1])

# print("Fixed p =", p, " →  φ coefficients:", phi)
# print("Fixed b =", b, " →  η weights shape:", eta.shape)
