"""
Wavelet-based crash time-series demo
------------------------------------

This script:

1. Generates synthetic daily crash data and covariates (rain, speed variance).
2. Performs Wavelet Multi-Resolution Analysis (MRA) on crash counts.
3. Computes wavelet cross-correlation between crashes and rain.
4. Computes wavelet coherence and phase between crashes and rain.

Dependencies:
    pip install numpy pandas matplotlib scipy pywavelets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy import signal


# ============================================================
# 1. Synthetic data generation
# ============================================================

def generate_synthetic_crash_data(n_years=5, seed=42):
    np.random.seed(seed)

    n_days = 365 * n_years
    t = np.arange(n_days)
    dates = pd.date_range(start="2015-01-01", periods=n_days, freq="D")

    # Long-term trend: gradual safety improvement (declining crash risk)
    trend = 50 - 0.01 * t  # baseline crashes slowly decreasing

    # Seasonal component: annual cycle (e.g., winter more hazardous)
    annual = 5 * np.sin(2 * np.pi * t / 365.0)

    # Weekly component: weekend peaks
    weekly = 3 * np.sin(2 * np.pi * t / 7.0)

    # Random weather "rain index": bursts clustered in some seasons
    rain_base = np.maximum(0, np.sin(2 * np.pi * (t - 60) / 365.0))  # wetter in late autumn/winter
    rain_noise = 0.5 * np.random.randn(n_days)
    rain_index = np.clip(rain_base + rain_noise, 0, None)

    # Speed variance: more unstable traffic during certain periods
    speed_variance = 1 + 0.2 * np.sin(2 * np.pi * t / 30.0)  # ~monthly pattern
    speed_variance += 0.3 * np.random.randn(n_days)

    # Crash intensity: combination of trend, seasonality, and exogenous drivers
    lambda_crash = (
        trend
        + annual
        + weekly
        + 8 * rain_index        # rain increases crashes
        + 2 * speed_variance    # unstable speed increases crashes
    )

    # Ensure positive rates
    lambda_crash = np.clip(lambda_crash, 5, None)

    # Realized crash counts (Poisson)
    crashes = np.random.poisson(lam=lambda_crash)

    df = pd.DataFrame({
        "date": dates,
        "crashes": crashes,
        "rain_index": rain_index,
        "speed_variance": speed_variance
    }).set_index("date")

    return df


# ============================================================
# 2. Wavelet helpers (with length fix)
# ============================================================

def reconstruct_component(coeffs, wavelet, level, orig_len,
                          component_type="approx", component_level=None):
    """
    Reconstruct a single component (approximation or detail) and
    force output length to `orig_len` to avoid dimension mismatches.

    component_type: "approx" or "detail"
    component_level:
        - if approx: ignored (we only use highest-level approx coeffs[0])
        - if detail: which detail level (1..level), where 1 is finest
    """
    coeffs_copy = [np.zeros_like(c) for c in coeffs]

    if component_type == "approx":
        coeffs_copy[0] = coeffs[0]
    elif component_type == "detail":
        # PyWavelets stores details as:
        #   coeffs[1] -> detail at level=level (coarsest)
        #   ...
        #   coeffs[level] -> detail at level=1 (finest)
        if component_level is None:
            raise ValueError("component_level must be specified for detail.")
        idx = level - component_level + 1
        coeffs_copy[idx] = coeffs[idx]
    else:
        raise ValueError("component_type must be 'approx' or 'detail'")

    rec = pywt.waverec(coeffs_copy, wavelet)

    # Fix length: trim or pad to match original length
    if len(rec) > orig_len:
        rec = rec[:orig_len]
    elif len(rec) < orig_len:
        # Pad with edge values to keep continuity
        pad_len = orig_len - len(rec)
        rec = np.pad(rec, (0, pad_len), mode="edge")

    return rec


def wavelet_details(series, wavelet, level):
    """
    Return detail components D1..D_level as full-length arrays,
    each forced to the original series length.
    """
    orig_len = len(series)
    coeffs = pywt.wavedec(series, wavelet, level=level)
    details = []
    for j in range(1, level + 1):
        d = reconstruct_component(
            coeffs, wavelet, level, orig_len,
            component_type="detail", component_level=j
        )
        details.append(d)
    return details, coeffs


def cross_corr(x, y, max_lag):
    """
    Normalized cross-correlation between two 1D series for lags in [-max_lag, max_lag].
    Returns lags array and correlation array.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    corr_full = signal.correlate(y, x, mode="full") / len(x)
    lags_full = signal.correlation_lags(len(y), len(x), mode="full")
    mask = (lags_full >= -max_lag) & (lags_full <= max_lag)
    return lags_full[mask], corr_full[mask]


def smooth_wavelet(power, time_smooth=5, scale_smooth=3):
    """
    Simple 2D smoothing operator for wavelet power/coherence:
    uniform filter across time and scale.
    """
    from scipy.ndimage import uniform_filter
    return uniform_filter(power.real, size=(scale_smooth, time_smooth))


# ============================================================
# 3. Main analysis routine
# ============================================================

def main():
    # -------------------------
    # 3.1 Generate data
    # -------------------------
    df = generate_synthetic_crash_data(n_years=5, seed=42)

    # Quick look
    print(df.head())

    # Plot raw series
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df.index, df["crashes"])
    axes[0].set_ylabel("Crashes")
    axes[0].set_title("Synthetic daily crashes")

    axes[1].plot(df.index, df["rain_index"])
    axes[1].set_ylabel("Rain index")

    axes[2].plot(df.index, df["speed_variance"])
    axes[2].set_ylabel("Speed variance")

    plt.tight_layout()
    plt.show()

    # -------------------------
    # 3.2 Wavelet MRA on crash counts
    # -------------------------
    series = df["crashes"].values
    orig_len = len(series)

    wavelet_name = "db4"
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(len(series), wavelet.dec_len)
    level = min(7, max_level)  # cap at 7 levels for interpretation

    coeffs = pywt.wavedec(series, wavelet_name, level=level)
    print("MRA levels:", level)
    print("Coefficient shapes:", [c.shape for c in coeffs])

    # Reconstruct approximation and chosen details with length fix
    approx = reconstruct_component(
        coeffs, wavelet_name, level, orig_len,
        component_type="approx"
    )
    detail_fine = reconstruct_component(
        coeffs, wavelet_name, level, orig_len,
        component_type="detail", component_level=1
    )
    detail_weekly = reconstruct_component(
        coeffs, wavelet_name, level, orig_len,
        component_type="detail", component_level=3
    )
    detail_seasonal = reconstruct_component(
        coeffs, wavelet_name, level, orig_len,
        component_type="detail", component_level=5
    )

    # Plot MRA components
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(df.index, series)
    axes[0].set_ylabel("Crashes")
    axes[0].set_title("Original crash counts")

    axes[1].plot(df.index, approx)
    axes[1].set_ylabel("Approx.")
    axes[1].set_title("Long-term trend (Approximation)")

    axes[2].plot(df.index, detail_seasonal)
    axes[2].set_ylabel("Detail")
    axes[2].set_title("Seasonal / low-frequency detail")

    axes[3].plot(df.index, detail_fine)
    axes[3].set_ylabel("Detail")
    axes[3].set_title("High-frequency detail (anomalies)")

    plt.tight_layout()
    plt.show()

    # -------------------------
    # 3.3 Wavelet cross-correlation (crashes vs. rain)
    # -------------------------
    crash_details, _ = wavelet_details(series, wavelet_name, level=level)
    rain_details, _ = wavelet_details(df["rain_index"].values, wavelet_name, level=level)

    max_lag = 30  # days
    scale_to_inspect = 3  # example: medium-scale detail

    d_crash = crash_details[scale_to_inspect - 1]
    d_rain = rain_details[scale_to_inspect - 1]

    lags, corr = cross_corr(d_crash, d_rain, max_lag)

    plt.figure(figsize=(8, 4))
    plt.stem(lags, corr)
    plt.xlabel("Lag (days) [rain_index leads if lag > 0]")
    plt.ylabel("Correlation")
    plt.title(f"Wavelet cross-correlation at detail level {scale_to_inspect}")
    plt.grid(True)
    plt.show()

    lag_peak = lags[np.argmax(corr)]
    corr_peak = corr.max()
    print(f"Peak cross-correlation at detail level {scale_to_inspect}:")
    print(f"  lag = {lag_peak} days, corr = {corr_peak:.3f}")

    # -------------------------
    # 3.4 Wavelet coherence & phase (crashes vs. rain)
    # -------------------------
    x = series.astype(float)
    y = df["rain_index"].values.astype(float)

    dt = 1.0  # daily sampling
    min_period = 2
    max_period = 128
    num_scales = 64
    periods = np.linspace(min_period, max_period, num_scales)
    scales = periods / dt

    wavelet_cwt = "cmor1.5-1.0"  # complex Morlet

    Wx, _ = pywt.cwt(x, scales, wavelet_cwt, sampling_period=dt)
    Wy, _ = pywt.cwt(y, scales, wavelet_cwt, sampling_period=dt)

    # Cross-wavelet transform
    Wxy = Wx * np.conjugate(Wy)

    # Smoothing
    Sxx = smooth_wavelet(np.abs(Wx) ** 2)
    Syy = smooth_wavelet(np.abs(Wy) ** 2)
    Sxy = smooth_wavelet(Wxy)

    # Coherence
    WCOH = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-10)

    # Phase of cross-wavelet
    phase = np.angle(Sxy)

    # Plot coherence
    fig, ax = plt.subplots(figsize=(12, 6))
    t_vals = np.arange(len(x))

    im = ax.imshow(
        WCOH,
        extent=[t_vals[0], t_vals[-1], periods[-1], periods[0]],
        aspect="auto"
    )

    ax.set_ylabel("Period (days)")
    ax.set_xlabel("Time index (days from start)")
    ax.set_title("Wavelet Coherence: crashes vs. rain_index")
    plt.colorbar(im, ax=ax, label="Coherence")
    plt.show()

    # Phase arrows (subsampled)
    step_t = 60   # time step
    step_s = 6    # scale step

    t_idx = np.arange(0, len(t_vals), step_t)
    s_idx = np.arange(0, len(periods), step_s)

    T, S = np.meshgrid(t_idx, periods[s_idx])
    phase_sub = phase[np.ix_(s_idx, t_idx)]
    U = np.cos(phase_sub)
    V = np.sin(phase_sub)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        WCOH,
        extent=[t_vals[0], t_vals[-1], periods[-1], periods[0]],
        aspect="auto"
    )
    # FIXED: use S (2D meshgrid) instead of periods[s_idx] (1D)
    ax.quiver(
        T, S, U, V,
        scale=40
    )

    ax.set_ylabel("Period (days)")
    ax.set_xlabel("Time index (days from start)")
    ax.set_title("Wavelet Coherence with Phase: crashes vs. rain_index")
    plt.colorbar(im, ax=ax, label="Coherence")
    plt.show()


if __name__ == "__main__":
    main()
