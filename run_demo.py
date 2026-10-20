"""Runs the radar tracking experiment and saves the charts in docs/.

    python run_demo.py

1. One example run, plotted on a map.
2. 500 Monte Carlo runs (same flight path, fresh radar noise each time) to measure
   how much the filter reduces position error compared with the raw detections.
3. A sweep of the filter's process-noise setting, to show the tuning trade-off.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tracker import RadarConfig, RadarEKF, drone_trajectory, polar_to_cartesian, radar_measurements

DOCS = Path(__file__).parent / "docs"
N_RUNS = 500
WARM_UP_SCANS = 5            # ignore the first few scans while the track settles
ACCEL_NOISE_STD = 1.0        # chosen from the tuning sweep below
TURN_START, TURN_END = 40, 70  # seconds, from the trajectory definition

# Chart styling
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
TRACK = "#2a78d6"            # filter estimate (blue)
DETECTION = "#eb6834"        # raw radar detections (orange)
TRUTH = "#52514e"            # true path (neutral)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED, "axes.titlecolor": TEXT,
    "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
})


def track_once(states, config, rng, accel_noise_std=ACCEL_NOISE_STD):
    measurements = radar_measurements(states, config, rng)
    ekf = RadarEKF(config.range_noise_std, config.bearing_noise_std, accel_noise_std)
    estimates, covariances = ekf.run(measurements, config.scan_period)
    return measurements, estimates, covariances


def errors(states, measurements, estimates):
    detected = ~np.isnan(measurements[:, 0])
    raw_xy = polar_to_cartesian(measurements[:, 0], measurements[:, 1])
    raw_err = np.linalg.norm(raw_xy - states[:, :2], axis=1)          # NaN where missed
    filt_err = np.linalg.norm(estimates[:, :2] - states[:, :2], axis=1)
    vel_err = np.linalg.norm(estimates[:, 2:] - states[:, 2:], axis=1)
    return detected, raw_err, filt_err, vel_err


def rms(values):
    values = np.asarray(values)
    return float(np.sqrt(np.nanmean(values ** 2)))


def monte_carlo(times, states, config, accel_noise_std, n_runs=N_RUNS):
    raw_all, filt_all, vel_all = [], [], []
    for seed in range(n_runs):
        rng = np.random.default_rng(seed)
        measurements, estimates, _ = track_once(states, config, rng, accel_noise_std)
        detected, raw_err, filt_err, vel_err = errors(states, measurements, estimates)
        # Compare like with like: only scans where the radar actually reported the target
        filt_err = np.where(detected, filt_err, np.nan)
        raw_all.append(raw_err)
        filt_all.append(filt_err)
        vel_all.append(vel_err)
    return np.array(raw_all), np.array(filt_all), np.array(vel_all)


def plot_map(times, states, measurements, estimates, path):
    raw_xy = polar_to_cartesian(measurements[:, 0], measurements[:, 1])
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})

    # Full picture
    ax_full.plot(states[:, 0], states[:, 1], color=TRUTH, lw=2, ls="--", label="True path")
    ax_full.plot(estimates[:, 0], estimates[:, 1], color=TRACK, lw=2, label="Kalman filter track")
    ax_full.plot(0, 0, marker="^", color=TEXT, ms=10, ls="none")
    ax_full.annotate("Radar", (0, 0), xytext=(10, -4), textcoords="offset points", color=TEXT, fontsize=10)
    ax_full.annotate("Start", states[0, :2], xytext=(0, -16), textcoords="offset points",
                     color=TEXT_MUTED, fontsize=9, ha="center")
    ax_full.annotate("End", states[-1, :2], xytext=(10, -4), textcoords="offset points",
                     color=TEXT_MUTED, fontsize=9)
    ax_full.set_xlim(-1200, 900)
    ax_full.set_ylim(-250, 3150)
    ax_full.set_aspect("equal")
    ax_full.set_xlabel("x (m)")
    ax_full.set_ylabel("y (m)")
    ax_full.set_title("Tracking a drone through a 90° turn")
    ax_full.legend(loc="upper left")

    # Zoom on a straight stretch, stretched vertically so the noise is visible
    x0, x1, y0, y1 = -720, -480, 1470, 1530
    ax_full.add_patch(plt.Rectangle((x0, y0 - 60), x1 - x0, (y1 - y0) + 120,
                                    fill=False, ec=TEXT, lw=1))
    ax_full.annotate("Close-up", (x0, y0 - 60), xytext=(0, -14), textcoords="offset points",
                     color=TEXT_MUTED, fontsize=9)

    ax_zoom.plot(states[:, 0], states[:, 1], color=TRUTH, lw=2, ls="--", label="True path")
    ax_zoom.plot(raw_xy[:, 0], raw_xy[:, 1], "o", color=DETECTION, ms=7,
                 mec=SURFACE, mew=1, label="Raw radar detections")
    ax_zoom.plot(estimates[:, 0], estimates[:, 1], color=TRACK, lw=2,
                 marker="o", ms=4, label="Kalman filter track")
    ax_zoom.set_xlim(x0, x1)
    ax_zoom.set_ylim(y0, y1)
    ax_zoom.set_xlabel("x (m)")
    ax_zoom.set_ylabel("y (m)  (stretched to show the noise)")
    ax_zoom.set_title("Close-up: noisy detections vs filtered track")
    ax_zoom.legend(loc="upper right", ncols=3, fontsize=9, bbox_to_anchor=(1, 1.0))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_over_time(times, raw_all, filt_all, path):
    raw_rmse = np.sqrt(np.nanmean(raw_all ** 2, axis=0))
    filt_rmse = np.sqrt(np.nanmean(filt_all ** 2, axis=0))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(TURN_START, TURN_END, color=GRID, alpha=0.6, lw=0)
    ax.text((TURN_START + TURN_END) / 2, 0.97, "Drone turning", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=TEXT_MUTED, fontsize=9)
    ax.plot(times, raw_rmse, color=DETECTION, lw=2, label="Raw radar detections")
    ax.plot(times, filt_rmse, color=TRACK, lw=2, label="Kalman filter")
    ax.text(times[-1] + 1, raw_rmse[-1], "Raw", color=TEXT_MUTED, va="center", fontsize=9)
    ax.text(times[-1] + 1, filt_rmse[-1], "Filter", color=TEXT_MUTED, va="center", fontsize=9)
    ax.set_xlim(0, times[-1] + 8)
    ax.set_ylim(0, None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error, RMS (m)")
    ax.set_title(f"Position error over time, averaged over {N_RUNS} simulated runs")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tuning(sweep, raw_baseline, path):
    accels = [a for a, _ in sweep]
    filt = [f for _, f in sweep]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(raw_baseline, color=DETECTION, lw=2, ls="--")
    ax.text(accels[0], raw_baseline, "  Raw detections", color=TEXT_MUTED, va="bottom", fontsize=9)
    ax.plot(accels, filt, color=TRACK, lw=2, marker="o", ms=6)
    best = int(np.argmin(filt))
    ax.annotate(f"Best: {accels[best]:g} m/s²", (accels[best], filt[best]), xytext=(0, -18),
                textcoords="offset points", ha="center", color=TEXT, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(accels)
    ax.set_xticklabels([f"{a:g}" for a in accels])
    ax.set_ylim(0, max(raw_baseline, max(filt)) * 1.15)
    ax.set_xlabel("Process noise setting: allowed acceleration (m/s², log scale)")
    ax.set_ylabel("Position error, RMS (m)")
    ax.set_title("Tuning the filter: too smooth lags, too twitchy follows the noise")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    DOCS.mkdir(exist_ok=True)
    config = RadarConfig()
    times, states = drone_trajectory(config.scan_period)

    # 1. One example run
    measurements, estimates, _ = track_once(states, config, np.random.default_rng(42))
    plot_map(times, states, measurements, estimates, DOCS / "tracking.png")

    # 2. Monte Carlo
    raw_all, filt_all, vel_all = monte_carlo(times, states, config, ACCEL_NOISE_STD)
    settled = slice(WARM_UP_SCANS, None)
    raw_rmse = rms(raw_all[:, settled])
    filt_rmse = rms(filt_all[:, settled])
    vel_rmse = rms(vel_all[:, settled])
    turning = (times >= TURN_START) & (times <= TURN_END)
    straight = ~turning & (np.arange(len(times)) >= WARM_UP_SCANS)
    plot_error_over_time(times, raw_all, filt_all, DOCS / "error_over_time.png")

    # 3. Tuning sweep
    sweep = []
    for accel in [0.1, 0.25, 0.5, 1, 2, 4, 8]:
        _, f, _ = monte_carlo(times, states, config, accel, n_runs=200)
        sweep.append((accel, rms(f[:, settled])))
    plot_tuning(sweep, raw_rmse, DOCS / "tuning.png")

    results = {
        "runs": N_RUNS,
        "raw_position_rmse_m": round(raw_rmse, 2),
        "filter_position_rmse_m": round(filt_rmse, 2),
        "position_error_reduction_pct": round(100 * (1 - filt_rmse / raw_rmse), 1),
        "filter_velocity_rmse_mps": round(vel_rmse, 2),
        "filter_position_rmse_straight_m": round(rms(filt_all[:, straight]), 2),
        "filter_position_rmse_turning_m": round(rms(filt_all[:, turning]), 2),
        "raw_position_rmse_turning_m": round(rms(raw_all[:, turning]), 2),
        "tuning_sweep": [{"accel_noise_std": a, "filter_position_rmse_m": round(f, 2)} for a, f in sweep],
    }
    (DOCS / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
