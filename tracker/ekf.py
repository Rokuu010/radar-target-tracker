"""An extended Kalman filter (EKF) that tracks one target from radar range/bearing detections.

State vector: [x, y, vx, vy]
    position in metres and velocity in metres per second.

Each scan the filter does two things:
    1. Predict: move the estimate forward in time, assuming the target keeps
       roughly the same velocity. Uncertainty grows, because the target might
       have changed course.
    2. Update: if the radar detected the target, blend the prediction with the
       measurement. The Kalman gain decides how much to trust each one, based
       on how uncertain they are. Uncertainty shrinks.

It's "extended" because the radar measures range and bearing, which are not a
straight-line (linear) function of x and y. The EKF handles that by
linearising the measurement around the current estimate (the Jacobian below).
"""

import numpy as np


def wrap_angle(angle):
    """Wraps an angle into [-pi, pi). Needed because 179 deg and -179 deg are only 2 deg apart."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class RadarEKF:
    def __init__(self, range_noise_std: float, bearing_noise_std: float,
                 accel_noise_std: float = 1.0, initial_speed_std: float = 30.0):
        """
        Args:
            range_noise_std: radar range noise (metres).
            bearing_noise_std: radar bearing noise (radians).
            accel_noise_std: how much the target is allowed to accelerate between scans
                (m/s^2). Higher means the filter adapts faster to turns but smooths less.
            initial_speed_std: uncertainty in velocity when a track starts (m/s), since a
                single detection says nothing about speed or direction.
        """
        self.R = np.diag([range_noise_std ** 2, bearing_noise_std ** 2])
        self.accel_noise_std = accel_noise_std
        self.initial_speed_std = initial_speed_std
        self.x = None          # state estimate
        self.P = None          # state covariance (how uncertain the estimate is)

    # ---- Track start ----------------------------------------------------------------

    def initialise(self, measurement: np.ndarray) -> None:
        """Starts a track from the first detection: position from the radar, velocity unknown."""
        r, theta = measurement
        self.x = np.array([r * np.cos(theta), r * np.sin(theta), 0.0, 0.0])

        # Convert the radar's range/bearing noise into x/y uncertainty.
        # Range noise is fixed but bearing noise grows with distance, so the uncertainty
        # is an ellipse stretched across the line of sight.
        J = np.array([[np.cos(theta), -r * np.sin(theta)],
                      [np.sin(theta),  r * np.cos(theta)]])
        position_cov = J @ self.R @ J.T

        self.P = np.zeros((4, 4))
        self.P[:2, :2] = position_cov
        self.P[2:, 2:] = np.eye(2) * self.initial_speed_std ** 2

    # ---- Predict ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Moves the estimate forward dt seconds using a constant-velocity model."""
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)

        # Process noise: random, unknown acceleration between scans
        q = self.accel_noise_std ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        Q = q * np.array([[dt4 / 4, 0,       dt3 / 2, 0],
                          [0,       dt4 / 4, 0,       dt3 / 2],
                          [dt3 / 2, 0,       dt2,     0],
                          [0,       dt3 / 2, 0,       dt2]])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # ---- Update -----------------------------------------------------------------

    @staticmethod
    def measurement_function(x: np.ndarray) -> np.ndarray:
        """What the radar would measure if the target were exactly at state x."""
        px, py = x[0], x[1]
        return np.array([np.hypot(px, py), np.arctan2(py, px)])

    @staticmethod
    def measurement_jacobian(x: np.ndarray) -> np.ndarray:
        """How range and bearing change with small changes in x, y (the linearisation)."""
        px, py = x[0], x[1]
        r2 = px ** 2 + py ** 2
        r = np.sqrt(r2)
        return np.array([[px / r,   py / r,  0, 0],
                         [-py / r2, px / r2, 0, 0]])

    def update(self, measurement: np.ndarray) -> None:
        """Corrects the prediction using a radar detection [range, bearing]."""
        H = self.measurement_jacobian(self.x)

        innovation = measurement - self.measurement_function(self.x)   # measured minus expected
        innovation[1] = wrap_angle(innovation[1])

        S = H @ self.P @ H.T + self.R              # uncertainty of the innovation
        K = self.P @ H.T @ np.linalg.inv(S)        # Kalman gain

        self.x = self.x + K @ innovation

        # Joseph form: algebraically the same as (I - KH)P, but keeps P symmetric and
        # positive definite despite rounding errors, which matters in long-running trackers
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    # ---- Run a whole track --------------------------------------------------------

    def run(self, measurements: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Tracks through a sequence of scans. NaN rows are missed detections (predict only).

        Returns:
            estimates: shape (N, 4), the state estimate after each scan.
            covariances: shape (N, 4, 4), the uncertainty after each scan.
        """
        estimates, covariances = [], []
        for k, z in enumerate(measurements):
            if k == 0:
                self.initialise(z)
            else:
                self.predict(dt)
                if not np.any(np.isnan(z)):
                    self.update(z)
            estimates.append(self.x.copy())
            covariances.append(self.P.copy())
        return np.array(estimates), np.array(covariances)
