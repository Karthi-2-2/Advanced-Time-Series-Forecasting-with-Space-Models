import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return y, S

    def log_likelihood(self, y, S):
        return -0.5 * (np.log(np.linalg.det(S)) + y.T @ np.linalg.inv(S) @ y + np.log(2*np.pi))


def run_kalman_filter(y, F, H, Q, R, x0, P0):
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    xs, Ps, ll = [], [], 0

    for z in y:
        kf.predict()
        innov, S = kf.update(np.array([[z]]))
        ll += kf.log_likelihood(innov, S)
        xs.append(kf.x.copy())
        Ps.append(kf.P.copy())

    return np.array(xs), np.array(Ps), ll
