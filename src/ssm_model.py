import numpy as np

def build_structural_ssm(params, seasonal_period=12):
    q_level, q_trend, q_season, r_obs = np.exp(params)

    n_season = seasonal_period - 1
    dim = 2 + n_season

    F = np.zeros((dim, dim))
    F[0,0] = 1; F[0,1] = 1
    F[1,1] = 1
    F[2:,2:] = np.roll(np.eye(n_season), -1, axis=1)
    F[2,-1] = -1

    H = np.zeros((1, dim))
    H[0,0] = 1
    H[0,2] = 1

    Q = np.diag([q_level, q_trend] + [q_season]*n_season)
    R = np.array([[r_obs]])

    x0 = np.zeros((dim,1))
    P0 = np.eye(dim) * 10

    return F, H, Q, R, x0, P0

