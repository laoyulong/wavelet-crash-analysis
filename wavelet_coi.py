# -*- coding: utf-8 -*-
"""
wavelet_coi_spatial.py
----------------------

Spatial multiscale crash-risk demo with:
(A) Synthetic crash points on a city-like grid (inhomogeneous Poisson + hotspots)
(B) Synthetic corridor-like risk fields (anisotropic elongated structures)

Includes:
- 2D DWT (PyWavelets)
- 2D CWT-like multiscale analysis via LoG (Laplacian of Gaussian) energy maps
- Spatial COI masks (support-based and e-folding analog)
- Persistence maps across scales
- Static plots
- Interactive viewer (matplotlib widgets)

Dependencies:
    pip install numpy matplotlib scipy pywavelets
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.ndimage import gaussian_filter, gaussian_laplace

# Interactive widgets (matplotlib built-in)
from matplotlib.widgets import Slider, RadioButtons

# ============================================================
# Utilities
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)

def min_distance_to_boundary(nx, ny):
    """
    Distance (in pixels) to the nearest boundary for each cell in an (nx, ny) grid.
    Returns array d shape (nx, ny):
        d[i,j] = min(i, nx-1-i, j, ny-1-j)
    """
    i = np.arange(nx)[:, None]          # (nx, 1)
    j = np.arange(ny)[None, :]          # (1, ny)
    ii = np.repeat(i, ny, axis=1)       # (nx, ny)
    jj = np.repeat(j, nx, axis=0)       # (nx, ny)

    d = np.minimum.reduce([ii, (nx - 1) - ii, jj, (ny - 1) - jj])
    return d.astype(float)

def coi_mask_support(nx, ny, a_pixels, K=2.0):
    """
    Support-based COI: inside-COI where d(x,y) < K * a_pixels
    """
    d = min_distance_to_boundary(nx, ny)
    return d < (K * float(a_pixels))

def coi_mask_efold(nx, ny, a_pixels):
    """
    E-folding COI analog (Morlet convention): inside-COI where d(x,y) < sqrt(2) * a_pixels
    """
    d = min_distance_to_boundary(nx, ny)
    return d < (np.sqrt(2.0) * float(a_pixels))

def plot_field(Z, title, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(7, 6))
    plt.imshow(Z, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def apply_coi_mask(binary_map, coi_inside):
    return binary_map & (~coi_inside)

def persistence_map(binary_masks):
    P = np.zeros_like(binary_masks[0], dtype=int)
    for m in binary_masks:
        P += m.astype(int)
    return P

# ============================================================
# Scenario A: City-like points -> raster
# ============================================================

def city_intensity_field(nx=128, ny=128):
    x = np.linspace(-1, 1, nx)[:, None]
    y = np.linspace(-1, 1, ny)[None, :]

    baseline = 0.8 + 0.6 * np.exp(-2.5 * (x**2 + y**2))

    def bump(x0, y0, sx, sy, amp):
        return amp * np.exp(-((x - x0)**2 / (2*sx**2) + (y - y0)**2 / (2*sy**2)))

    lam = baseline
    lam += bump(-0.4,  0.2, 0.12, 0.10, 2.0)
    lam += bump( 0.5, -0.3, 0.10, 0.14, 1.8)
    lam += bump( 0.2,  0.5, 0.08, 0.08, 1.5)

    ring = np.exp(-((np.sqrt(x**2 + y**2) - 0.65)**2) / (2*(0.06**2)))
    lam += 0.5 * ring

    lam = np.clip(lam, 0, None)
    lam = lam / lam.mean()
    lam *= 1.2
    return lam

def sample_city_points_from_intensity(lam, seed=42):
    set_seed(seed)
    Z = np.random.poisson(lam=lam)

    pts = []
    nx, ny = Z.shape
    for i in range(nx):
        for j in range(ny):
            k = Z[i, j]
            if k > 0:
                xs = i + np.random.rand(k)
                ys = j + np.random.rand(k)
                pts.append(np.column_stack([xs, ys]))
    pts = np.vstack(pts) if len(pts) else np.zeros((0, 2))
    return Z.astype(float), pts

def point_raster_to_density(Z_counts, sigma=1.5):
    return gaussian_filter(Z_counts, sigma=sigma)

# ============================================================
# Scenario B: Corridor-like risk field
# ============================================================

def corridor_risk_field(nx=128, ny=128, seed=7):
    set_seed(seed)
    X = np.linspace(0, 1, nx)[:, None]
    Y = np.linspace(0, 1, ny)[None, :]

    def dist_to_line(a, b, c):
        return np.abs(a*X + b*Y + c) / np.sqrt(a*a + b*b + 1e-12)

    d1 = dist_to_line(0.8, -1.0, 0.2)  # diagonal corridor
    corr1 = np.exp(-(d1**2) / (2*(0.02**2)))
    taper1 = np.exp(-((X - 0.55)**2) / (2*(0.35**2)))

    d2 = np.abs(Y - 0.65)              # horizontal corridor
    corr2 = np.exp(-(d2**2) / (2*(0.018**2)))
    taper2 = np.exp(-((X - 0.45)**2) / (2*(0.40**2)))

    base = 0.4 + 0.6*np.exp(-((X-0.5)**2 + (Y-0.5)**2)/(2*(0.35**2)))
    Z = base + 2.5*corr1*taper1 + 2.0*corr2*taper2

    def blob(x0, y0, s, amp):
        return amp*np.exp(-(((X-x0)**2 + (Y-y0)**2)/(2*s*s)))

    Z += blob(0.22, 0.36, 0.020, 3.0)
    Z += blob(0.68, 0.72, 0.025, 2.5)
    Z += blob(0.58, 0.46, 0.018, 2.0)

    noise = gaussian_filter(np.random.randn(nx, ny), sigma=1.0)
    Z = Z + 0.25*noise

    Z = np.clip(Z, 0, None)
    return Z

# ============================================================
# 2D DWT
# ============================================================

def dwt2_decompose(Z, wavelet="db2", level=3):
    return pywt.wavedec2(Z, wavelet=wavelet, level=level, mode="periodization")

def dwt2_reconstruct_component(coeffs, wavelet, component="A", j=None, band=None):
    """
    Reconstruct approximation ("A") or a detail band ("D") at level j and band in {"LH","HL","HH"}.
    In PyWavelets details are (cH, cV, cD) which correspond roughly to (HL, LH, HH).
    """
    coeffs_zero = []
    cA = coeffs[0]
    coeffs_zero.append(np.zeros_like(cA))

    details = coeffs[1:]
    for (cH, cV, cD) in details:
        coeffs_zero.append((np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)))

    if component == "A":
        coeffs_zero[0] = coeffs[0]
    elif component == "D":
        if j is None or band is None:
            raise ValueError("Specify j (1..level) and band in {'LH','HL','HH'}")
        level = len(coeffs) - 1
        idx = level - j + 1
        cH, cV, cD = coeffs[idx]
        if band == "HL":
            coeffs_zero[idx] = (cH, np.zeros_like(cV), np.zeros_like(cD))
        elif band == "LH":
            coeffs_zero[idx] = (np.zeros_like(cH), cV, np.zeros_like(cD))
        elif band == "HH":
            coeffs_zero[idx] = (np.zeros_like(cH), np.zeros_like(cV), cD)
        else:
            raise ValueError("band must be one of {'LH','HL','HH'}")
    else:
        raise ValueError("component must be 'A' or 'D'")

    Zrec = pywt.waverec2(coeffs_zero, wavelet=wavelet, mode="periodization")
    return Zrec

# ============================================================
# 2D CWT-like multiscale (LoG)
# ============================================================

def log_multiscale_energy(Z, sigmas):
    responses = []
    energy = []
    for s in sigmas:
        r = gaussian_laplace(Z, sigma=float(s))
        responses.append(r)
        energy.append(r*r)
    return responses, energy

# ============================================================
# Surrogate thresholds (lightweight)
# ============================================================

def surrogate_threshold_city_poisson(lam, sigmas, M=40, q=0.95, smooth_sigma=1.5, seed=123):
    set_seed(seed)
    nx, ny = lam.shape
    energies = [np.zeros((M, nx, ny), dtype=float) for _ in sigmas]

    for m in range(M):
        Zm = np.random.poisson(lam=lam).astype(float)
        Zm = gaussian_filter(Zm, sigma=smooth_sigma)
        _, Em_list = log_multiscale_energy(Zm, sigmas)
        for k, Em in enumerate(Em_list):
            energies[k][m] = Em

    thresholds = []
    for k in range(len(sigmas)):
        thresholds.append(np.quantile(energies[k], q, axis=0))
    return thresholds

def surrogate_threshold_generic_gaussian(Z, sigmas, M=40, q=0.95, noise_sigma=0.35, seed=123):
    set_seed(seed)
    nx, ny = Z.shape
    energies = [np.zeros((M, nx, ny), dtype=float) for _ in sigmas]

    for m in range(M):
        Zm = Z + noise_sigma*np.random.randn(nx, ny)
        Zm = gaussian_filter(Zm, sigma=1.0)
        _, Em_list = log_multiscale_energy(Zm, sigmas)
        for k, Em in enumerate(Em_list):
            energies[k][m] = Em

    thresholds = []
    for k in range(len(sigmas)):
        thresholds.append(np.quantile(energies[k], q, axis=0))
    return thresholds

# ============================================================
# Static pipeline runner (city/corridor)
# ============================================================

def run_city_static(nx=128, ny=128, sigmas=(1,2,4,8), wavelet="db2", level=3, K_support=2.0):
    print("\n=== Scenario A (static): City-like grid from synthetic points ===")
    lam = city_intensity_field(nx, ny)
    Z_counts, _ = sample_city_points_from_intensity(lam, seed=42)
    Z_density = point_raster_to_density(Z_counts, sigma=1.5)

    plot_field(Z_counts, "City: crash counts per cell")
    plot_field(Z_density, "City: smoothed crash density")

    coeffs = dwt2_decompose(Z_density, wavelet=wavelet, level=level)
    plot_field(coeffs[0], f"City: 2D DWT approximation A_{level}")

    for band in ["HL","LH","HH"]:
        Zd = dwt2_reconstruct_component(coeffs, wavelet, component="D", j=2, band=band)
        plot_field(Zd, f"City: 2D DWT detail (j=2, band={band})")

    sigmas = list(sigmas)
    _, energy = log_multiscale_energy(Z_density, sigmas)
    thresholds = surrogate_threshold_city_poisson(lam, sigmas, M=40, q=0.95, smooth_sigma=1.5)

    sig_masks = []
    for s, E, thr in zip(sigmas, energy, thresholds):
        sig_map = E > thr
        coi_ef = coi_mask_efold(nx, ny, a_pixels=s)
        sig_map_masked = apply_coi_mask(sig_map, coi_ef)
        sig_masks.append(sig_map_masked)
        plot_field(sig_map_masked.astype(float),
                   f"City: significant energy (LoG sigma={s}) + COI(efold)",
                   cmap="magma", vmin=0, vmax=1)

    P = persistence_map(sig_masks)
    plot_field(P, "City: persistence map (#scales significant, COI-masked)", cmap="inferno")

    return Z_density, sigmas, energy, thresholds

def run_corridor_static(nx=128, ny=128, sigmas=(1,2,4,8), wavelet="db2", level=3, K_support=2.0):
    print("\n=== Scenario B (static): Corridor-like synthetic risk field ===")
    Z = corridor_risk_field(nx, ny, seed=7)
    Z = gaussian_filter(Z, sigma=0.8)

    plot_field(Z, "Corridor: synthetic risk field")

    coeffs = dwt2_decompose(Z, wavelet=wavelet, level=level)
    plot_field(coeffs[0], f"Corridor: 2D DWT approximation A_{level}")

    for band in ["HL","LH","HH"]:
        Zd = dwt2_reconstruct_component(coeffs, wavelet, component="D", j=2, band=band)
        plot_field(Zd, f"Corridor: 2D DWT detail (j=2, band={band})")

    sigmas = list(sigmas)
    _, energy = log_multiscale_energy(Z, sigmas)
    thresholds = surrogate_threshold_generic_gaussian(Z, sigmas, M=40, q=0.95, noise_sigma=0.35)

    sig_masks = []
    for s, E, thr in zip(sigmas, energy, thresholds):
        sig_map = E > thr
        coi_sup = coi_mask_support(nx, ny, a_pixels=s, K=K_support)
        sig_map_masked = apply_coi_mask(sig_map, coi_sup)
        sig_masks.append(sig_map_masked)
        plot_field(sig_map_masked.astype(float),
                   f"Corridor: significant energy (LoG sigma={s}) + COI(support)",
                   cmap="magma", vmin=0, vmax=1)

    P = persistence_map(sig_masks)
    plot_field(P, "Corridor: persistence map (#scales significant, COI-masked)", cmap="inferno")

    return Z, sigmas, energy, thresholds

# ============================================================
# Interactive viewer
# ============================================================

def interactive_viewer(Z, sigmas, energy, thresholds, title_prefix="Viewer", K_support=2.0):
    """
    Interactive display with:
    - Scale slider (select sigma index)
    - COI type (support / efold)
    - Overlay mode: Energy / Significant / Persistence
    - Threshold quantile slider (approx; rescales thresholds linearly for demo)
      NOTE: For real work, recompute thresholds per q. Here we provide a pragmatic viewer.
    """
    nx, ny = Z.shape
    sigmas = list(sigmas)

    # Precompute persistence under default COI+threshold
    def compute_sig_masks(coi_type="efold", q_scale=1.0):
        masks = []
        for s, E, thr in zip(sigmas, energy, thresholds):
            thr_eff = thr * q_scale
            sig_map = E > thr_eff
            if coi_type == "support":
                coi = coi_mask_support(nx, ny, a_pixels=s, K=K_support)
            else:
                coi = coi_mask_efold(nx, ny, a_pixels=s)
            masks.append(apply_coi_mask(sig_map, coi))
        return masks

    # Initial state
    coi_type = "efold"
    overlay = "Energy"
    q_scale = 1.0
    sig_masks = compute_sig_masks(coi_type=coi_type, q_scale=q_scale)
    P = persistence_map(sig_masks)

    # Figure layout
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    idx0 = 0
    im = ax.imshow(energy[idx0], origin="lower", cmap="viridis")
    ax.set_title(f"{title_prefix}: {overlay} (sigma={sigmas[idx0]})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Slider for scale index
    ax_scale = plt.axes([0.25, 0.12, 0.65, 0.03])
    sld = Slider(ax_scale, "Scale idx", 0, len(sigmas)-1, valinit=0, valstep=1)

    # Radio buttons for overlay
    ax_overlay = plt.axes([0.02, 0.55, 0.18, 0.18])
    radio_overlay = RadioButtons(ax_overlay, ("Energy", "Significant", "Persistence"), active=0)

    # Radio buttons for COI type
    ax_coi = plt.axes([0.02, 0.78, 0.18, 0.14])
    radio_coi = RadioButtons(ax_coi, ("efold", "support"), active=0)

    # Threshold scaling slider (viewer convenience)
    ax_q = plt.axes([0.25, 0.06, 0.65, 0.03])
    sld_q = Slider(ax_q, "Thr scale", 0.5, 1.5, valinit=1.0, valstep=0.01)

    def redraw(*_):
        nonlocal sig_masks, P, coi_type, overlay, q_scale

        idx = int(sld.val)
        coi_type = radio_coi.value_selected
        overlay = radio_overlay.value_selected
        q_scale = float(sld_q.val)

        sig_masks = compute_sig_masks(coi_type=coi_type, q_scale=q_scale)
        P = persistence_map(sig_masks)

        if overlay == "Energy":
            data = energy[idx]
            im.set_cmap("viridis")
        elif overlay == "Significant":
            data = sig_masks[idx].astype(float)
            im.set_cmap("magma")
        else:  # Persistence
            data = P.astype(float)
            im.set_cmap("inferno")

        im.set_data(data)
        im.set_clim(vmin=np.nanmin(data), vmax=np.nanmax(data) + 1e-12)

        ax.set_title(f"{title_prefix}: {overlay} (sigma={sigmas[idx]}), COI={coi_type}, ThrScale={q_scale:.2f}")
        fig.canvas.draw_idle()

    sld.on_changed(redraw)
    sld_q.on_changed(redraw)
    radio_overlay.on_clicked(redraw)
    radio_coi.on_clicked(redraw)

    plt.show()

# ============================================================
# Main
# ============================================================

def main():
    # --- Static runs (thesis-friendly figures)
    Z_city, sigmas_city, energy_city, thr_city = run_city_static()
    Z_cor,  sigmas_cor,  energy_cor,  thr_cor  = run_corridor_static()

    # --- Interactive viewers
    interactive_viewer(Z_city, sigmas_city, energy_city, thr_city, title_prefix="City", K_support=2.0)
    interactive_viewer(Z_cor,  sigmas_cor,  energy_cor,  thr_cor,  title_prefix="Corridor", K_support=2.0)

    print("\nDone.")
    print("If you want thesis-grade threshold control:")
    print(" - recompute surrogate thresholds for each chosen quantile q (instead of Thr scale).")
    print(" - add connected-component labeling + hotspot ranking (area, centroid, elongation).")
    print(" - add FDR per scale if you compute p-values rather than quantile thresholds.")

if __name__ == "__main__":
    main()
