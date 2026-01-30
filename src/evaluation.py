import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalman import run_kalman_filter
from ssm_model import build_structural_ssm
from train import train_ssm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_kf(x_last, P_last, F, H, Q, steps):
    preds = []
    x, P = x_last, P_last
    for _ in range(steps):
        x = F @ x
        P = F @ P @ F.T + Q
        preds.append((H @ x).item())
    return np.array(preds)

if __name__ == "__main__":
    df = pd.read_csv("data/air_passengers.csv")
    y = df["value"].values
    train, test = y[:-12], y[-12:]

    params = train_ssm(train)
    F, H, Q, R, x0, P0 = build_structural_ssm(params)
    xs, Ps, _ = run_kalman_filter(train, F, H, Q, R, x0, P0)

    ssm_forecast = forecast_kf(xs[-1], Ps[-1], F, H, Q, 12)

    sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_forecast = sarima.fit().forecast(12)

    print("SSM RMSE:", np.sqrt(mean_squared_error(test, ssm_forecast)))
    print("SARIMA RMSE:", np.sqrt(mean_squared_error(test, sarima_forecast)))
    print("SSM MAE:", mean_absolute_error(test, ssm_forecast))
    print("SARIMA MAE:", mean_absolute_error(test, sarima_forecast))

    plt.plot(y, label="Actual")
    plt.plot(range(len(train), len(y)), ssm_forecast, label="SSM Forecast")
    plt.plot(range(len(train), len(y)), sarima_forecast, label="SARIMA Forecast")
    plt.legend()
    plt.show()
