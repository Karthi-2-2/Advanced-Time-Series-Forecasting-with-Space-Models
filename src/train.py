import numpy as np
import pandas as pd
from scipy.optimize import minimize
from kalman import run_kalman_filter
from ssm_model import build_structural_ssm

def neg_loglike(params, y):
    F, H, Q, R, x0, P0 = build_structural_ssm(params)
    _, _, ll = run_kalman_filter(y, F, H, Q, R, x0, P0)
    return -ll

def train_ssm(y):
    init_params = np.log([0.1, 0.01, 0.1, 1.0])
    res = minimize(neg_loglike, init_params, args=(y,), method="L-BFGS-B")
    return res.x

if __name__ == "__main__":
    df = pd.read_csv("data/air_passengers.csv")
    y = df["value"].values
    params = train_ssm(y)
    print("Optimized parameters:", np.exp(params))
