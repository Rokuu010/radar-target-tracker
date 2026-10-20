"""Simulates a drone's flight path and what a 2D surveillance radar would report about it.

The radar sits at the origin (0, 0). Once per scan it reports the target's
range (distance, in metres) and bearing (angle from the x-axis, in radians),
each with random noise. Sometimes it misses the target entirely.
"""

from dataclasses import dataclass

import numpy as np

from .ekf import wrap_angle


@dataclass
class RadarConfig:
    scan_period: float = 1.0                     # seconds between scans
    range_noise_std: float = 5.0                 # metres
    bearing_noise_std: float = np.deg2rad(0.3)   # radians
    detection_probability: float = 0.9           # chance of detecting the target on each scan


def drone_trajectory(scan_period: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth path of a drone: a straight leg, a 90-degree turn, then another straight leg.

    Returns:
        times: shape (N,), seconds since the start.
        states: shape (N, 4), each row is [x, y, vx, vy] in metres and metres per second.
    """
    speed = 20.0                        # m/s, a typical small drone
    turn_rate = np.deg2rad(3.0)         # rad/s, so a 90-degree turn takes 30 s
    legs = [(40, 0.0), (30, turn_rate), (50, 0.0)]   # (duration in s, turn rate)

    position = np.array([-1000.0, 1500.0])
    heading = 0.0                       # flying along +x to start with
    dt = scan_period

    times, states = [0.0], [[*position, speed * np.cos(heading), speed * np.sin(heading)]]
    t = 0.0
    for duration, omega in legs:
        for _ in range(int(round(duration / dt))):
            if omega == 0.0:
                position = position + speed * dt * np.array([np.cos(heading), np.sin(heading)])
            else:
                # Exact motion along an arc for a constant-rate turn
                new_heading = heading + omega * dt
                position = position + (speed / omega) * np.array([
                    np.sin(new_heading) - np.sin(heading),
                    np.cos(heading) - np.cos(new_heading),
                ])
                heading = new_heading
            t += dt
            times.append(t)
            states.append([*position, speed * np.cos(heading), speed * np.sin(heading)])

    return np.array(times), np.array(states)


def radar_measurements(states: np.ndarray, config: RadarConfig, rng: np.random.Generator):
    """Turns true target states into noisy radar detections.

    Returns:
        measurements: shape (N, 2), each row is [range, bearing]. Rows are NaN on scans
            where the radar missed the target.
    """
    x, y = states[:, 0], states[:, 1]
    true_range = np.hypot(x, y)
    true_bearing = np.arctan2(y, x)

    n = len(states)
    measured_range = true_range + rng.normal(0.0, config.range_noise_std, n)
    # Like a real radar, report bearings in [-180, 180) degrees
    measured_bearing = wrap_angle(true_bearing + rng.normal(0.0, config.bearing_noise_std, n))

    measurements = np.column_stack([measured_range, measured_bearing])
    missed = rng.random(n) > config.detection_probability
    missed[0] = False                   # always detect on the first scan so a track can start
    measurements[missed] = np.nan
    return measurements


def polar_to_cartesian(range_m: np.ndarray, bearing: np.ndarray) -> np.ndarray:
    """Converts radar (range, bearing) to (x, y) positions."""
    return np.column_stack([range_m * np.cos(bearing), range_m * np.sin(bearing)])
