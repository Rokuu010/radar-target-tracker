import numpy as np
import pytest

from tracker import RadarConfig, RadarEKF, polar_to_cartesian, radar_measurements, wrap_angle


def straight_line(start, velocity, n_scans=60, dt=1.0):
    """Ground truth for a target flying in a straight line at constant velocity."""
    t = np.arange(n_scans) * dt
    positions = np.asarray(start) + np.outer(t, velocity)
    velocities = np.tile(velocity, (n_scans, 1))
    return np.hstack([positions, velocities])


def run_filter(states, config, seed):
    measurements = radar_measurements(states, config, np.random.default_rng(seed))
    ekf = RadarEKF(config.range_noise_std, config.bearing_noise_std, accel_noise_std=1.0)
    estimates, covariances = ekf.run(measurements, config.scan_period)
    return measurements, estimates, covariances


def test_wrap_angle_handles_the_plus_minus_180_degree_seam():
    assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
    assert wrap_angle(0.5) == pytest.approx(0.5)


def test_filter_is_more_accurate_than_raw_detections():
    config = RadarConfig()
    states = straight_line(start=[-1500, 2000], velocity=[20, 0])
    raw_sq, filt_sq = [], []
    for seed in range(50):
        measurements, estimates, _ = run_filter(states, config, seed)
        detected = ~np.isnan(measurements[:, 0])
        detected[:5] = False                       # skip the warm-up scans
        raw_xy = polar_to_cartesian(measurements[:, 0], measurements[:, 1])
        raw_sq.append(np.sum((raw_xy[detected] - states[detected, :2]) ** 2, axis=1))
        filt_sq.append(np.sum((estimates[detected, :2] - states[detected, :2]) ** 2, axis=1))
    raw_rmse = np.sqrt(np.mean(np.concatenate(raw_sq)))
    filt_rmse = np.sqrt(np.mean(np.concatenate(filt_sq)))
    assert filt_rmse < 0.8 * raw_rmse


def test_filter_estimates_velocity_that_the_radar_never_measures():
    config = RadarConfig()
    true_velocity = np.array([15.0, -10.0])
    states = straight_line(start=[500, 2500], velocity=true_velocity)
    final_velocities = [run_filter(states, config, seed)[1][-1, 2:] for seed in range(30)]
    assert np.mean(final_velocities, axis=0) == pytest.approx(true_velocity, abs=1.5)


def test_uncertainty_grows_when_predicting_and_shrinks_when_updating():
    config = RadarConfig()
    ekf = RadarEKF(config.range_noise_std, config.bearing_noise_std)
    ekf.initialise(np.array([2000.0, 0.5]))
    ekf.predict(1.0)
    ekf.update(np.array([2000.0, 0.5]))
    before = np.trace(ekf.P)
    ekf.predict(1.0)
    after_predict = np.trace(ekf.P)
    ekf.update(np.array([2000.0, 0.5]))
    after_update = np.trace(ekf.P)
    assert after_predict > before
    assert after_update < after_predict


def test_missed_detections_are_bridged_by_prediction():
    config = RadarConfig(detection_probability=1.0)
    states = straight_line(start=[1000, 1000], velocity=[10, 10], n_scans=30)
    measurements = radar_measurements(states, config, np.random.default_rng(0))
    measurements[20:25] = np.nan                    # radar loses the target for 5 scans
    ekf = RadarEKF(config.range_noise_std, config.bearing_noise_std)
    estimates, covariances = ekf.run(measurements, config.scan_period)
    gap_error = np.linalg.norm(estimates[24, :2] - states[24, :2])
    assert gap_error < 30                           # still close after coasting for 5 s
    assert np.trace(covariances[24]) > np.trace(covariances[19])   # and knows it's less sure


def test_update_across_the_180_degree_seam_is_a_small_correction():
    # The estimate is at bearing +179.94 deg and the radar reports -179.94 deg, which is
    # only 0.11 deg away. Without angle wrapping the filter would see a
    # ~360 deg error and throw the estimate across the map.
    config = RadarConfig()
    ekf = RadarEKF(config.range_noise_std, config.bearing_noise_std)
    ekf.initialise(np.array([2000.0, np.pi - 0.001]))
    before = ekf.x[:2].copy()
    ekf.update(np.array([2000.0, -np.pi + 0.001]))
    assert np.linalg.norm(ekf.x[:2] - before) < 5


def test_tracks_a_target_crossing_the_180_degree_bearing():
    # Flying straight across the negative x-axis, where the bearing jumps from +180 to -180 deg.
    config = RadarConfig()
    states = straight_line(start=[-2000, 400], velocity=[0, -20], n_scans=40)
    for seed in range(20):
        _, estimates, _ = run_filter(states, config, seed)
        errors = np.linalg.norm(estimates[5:, :2] - states[5:, :2], axis=1)
        assert errors.max() < 50


def test_covariance_stays_symmetric_and_positive_definite():
    config = RadarConfig()
    states = straight_line(start=[-1500, 2000], velocity=[20, 0], n_scans=200)
    _, _, covariances = run_filter(states, config, seed=3)
    for P in covariances:
        assert np.allclose(P, P.T)
        assert np.all(np.linalg.eigvalsh(P) > 0)
