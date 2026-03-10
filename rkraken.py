import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from math import floor
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import fftconvolve, find_peaks, peak_widths
from matplotlib import patheffects as pe
from skimage.restoration import denoise_tv_bregman
from scipy.ndimage import median_filter, gaussian_filter
from scipy.special import i0e, i1e
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import sqrtm
import glob
import os
from PIL import Image


# Reduced Planck constant in eV*fs (approx CODATA): ħ = 6.582119569e-16 eV·s
hbar = 6.582119569e-1

def CFT(T_range, signal, use_window=True, inverse=False, zero_pad=0):
    """
    Continuous-time Fourier transform convention per column (fixed energy E):
        Forward: S(ω, E) = ∫ s(t, E) · e^{-i ω t} dt

    Discrete implementation with FFT for samples t_n = t0 + n·dt:
        S(ω_k, E) ≈ dt · e^{-i ω_k t0} · FFT_n{ s(t_n, E) }[k]

    Options:
      - use_window: apply Hann window (coherent-gain normalized) along time (rows).
      - inverse: when True, perform the inverse operation to reconstruct s(t, E)
                 from S(ω, E). The input "signal" must be the shifted spectrum
                 (as returned by the forward path). Windowing is not undone.
      - zero_pad: integer number of zeros to pad on BOTH sides along time before
                  transform (forward) or to account for when inverting. Default 0.

    Returns (forward, inverse respectively):
      - forward: spec_shift, OM_T, E_min, E_max
      - inverse: sig_time (trimmed to len(T_range) along rows), None, t_min, t_max
    """
    # Validate input and infer sizes locally (avoid globals)
    t = np.asarray(T_range)
    if t.ndim != 1:
        raise ValueError("T_range must be a 1D array of time samples")
    NT_local = t.size
    NE_local = signal.shape[1]

    # Ensure uniform sampling
    if NT_local < 2:
        raise ValueError("T_range must contain at least two samples")
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=0.0):
        raise ValueError("T_range must be uniformly spaced")

    # Inverse path: reconstruct time-domain from shifted spectrum
    if inverse:
        # Infer padded length from provided spectrum along rows
        S_shift = np.asarray(signal)
        if S_shift.ndim != 2:
            raise ValueError("For inverse, 'signal' must be a 2D array with rows as ω bins")
        N_eff = S_shift.shape[0]

        # Frequency grid associated with the padded length
        freqs = np.fft.fftfreq(N_eff, d=dt)
        omega = 2 * np.pi * freqs

        # Undo shift and the forward phase factor (use padded start time)
        t0_eff = t[0] - int(zero_pad) * dt
        S_unshift = np.fft.ifftshift(S_shift, axes=0)
        phase_inv = np.exp(1j * omega[:, None] * t0_eff)
        S_unphased = S_unshift * phase_inv

        # Inverse scaling: x ≈ (1/dt) · IFFT{ S }
        sig_time_full = np.fft.ifft(S_unphased, axis=0) / dt

        # Trim padding back to original time length
        zp = int(zero_pad)
        if zp > 0:
            lo = zp
            hi = zp + NT_local
            sig_time = sig_time_full[lo:hi, :]
        else:
            sig_time = sig_time_full

        # Return time-domain signal and simple time metadata
        return sig_time, None, float(t[0]), float(t[-1])

    # Forward path
    if signal.shape[0] != NT_local:
        raise ValueError("signal shape[0] must match len(T_range)")

    # Windowing (apply on original length), then symmetric zero padding
    if use_window:
        window = np.hanning(NT_local)[:, None]
        coherent_gain = window.mean()  # Hann coherent gain = 0.5
    else:
        window = np.ones((NT_local, 1))
        coherent_gain = 1.0

    windowed = (signal * window)

    zp = int(zero_pad)
    if zp > 0:
        windowed = np.pad(windowed, ((zp, zp), (0, 0)), mode='constant', constant_values=0)
    N_eff = windowed.shape[0]

    # Frequencies (unshifted) and corresponding angular frequencies for padded length
    freqs = np.fft.fftfreq(N_eff, d=dt)   # cycles per unit time
    omega = 2 * np.pi * freqs             # rad / unit time

    # FFT along time axis with dt scaling to approximate the continuous integral
    spec = np.fft.fft(windowed, axis=0) * dt

    # Phase correction for start time including leading padding
    t0_eff = t[0] - zp * dt
    phase = np.exp(-1j * omega[:, None] * t0_eff)
    spec *= phase

    # Shift spectrum and build energy axis (ħ·ω)
    spec_shift = np.fft.fftshift(spec, axes=0)
    energy_axis = hbar * omega
    energy_axis_shift = np.fft.fftshift(energy_axis)

    # Trim back to original NT_local size (keep central frequency bins)
    if zp > 0:
        trim_lo = (N_eff - NT_local) // 2
        trim_hi = trim_lo + NT_local
        spec_shift = spec_shift[trim_lo:trim_hi, :]
        energy_axis_shift = energy_axis_shift[trim_lo:trim_hi]

    OM_T = (energy_axis_shift / hbar)  # back to angular frequency ω
    OM_T = np.tile(OM_T, (NE_local, 1)).T

    return spec_shift, OM_T, energy_axis_shift[0], energy_axis_shift[-1]

def _style_ax(ax):
    """Apply professional tick styling to an axes object."""
    _tick_kw = dict(fontsize=9)
    ax.tick_params(axis='both', which='major', direction='in', length=5, width=0.4,
                   top=True, right=True, labelsize=_tick_kw['fontsize'])
    # ax.tick_params(axis='both', which='minor', direction='in', length=2.5, width=0.5,
    #                top=True, right=True)
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def plot_mat(mat,extent=[0,1,0,1],cmap='plasma',mode='phase',show=True,saveloc=None,caption=None,xlabel='x',ylabel='y',title=None):

    _label_kw = dict(fontsize=10,)
    _title_kw = dict(fontsize=11, fontweight='semibold')

    if mode == 'abs':
        plt.figure()
        im = plt.imshow(np.abs(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        plt.xlabel(xlabel, **_label_kw)
        plt.ylabel(ylabel, **_label_kw)
        if title:
            plt.title(title, **_title_kw)
        else:
            plt.title('|M|', **_title_kw)
        plt.colorbar(im)
        if caption is not None:
            plt.text(0.02, 0.98, caption, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='left',
                    color='white', weight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        _style_ax(plt.gca())

    elif mode == 'phase':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        im0 = axes[0].imshow(np.abs(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        if title:
            axes[0].set_title(title, **_title_kw)
        else:
            axes[0].set_title('|M|', **_title_kw)
        axes[0].set_xlabel(xlabel, **_label_kw)
        axes[0].set_ylabel(ylabel, **_label_kw)
        fig.colorbar(im0, ax=axes[0])
        if caption is not None:
            axes[0].text(0.02, 0.98, caption, transform=axes[0].transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='left',
                        color='white', weight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        im1 = axes[1].imshow(np.angle(mat), extent=extent, origin='lower', aspect='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('arg(M)', **_title_kw)
        axes[1].set_xlabel(xlabel, **_label_kw)
        fig.colorbar(im1, ax=axes[1])

        if title:
            fig.suptitle(title, **_title_kw)

        for ax in axes:
            _style_ax(ax)
        plt.tight_layout()

    elif mode == 'reim':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        im0 = axes[0].imshow(np.real(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[0].set_title('Re{M}', **_title_kw)
        axes[0].set_xlabel(xlabel, **_label_kw)
        axes[0].set_ylabel(ylabel, **_label_kw)
        fig.colorbar(im0, ax=axes[0])
        if caption is not None:
            axes[0].text(0.02, 0.98, caption, transform=axes[0].transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='left',
                        color='white', weight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        im1 = axes[1].imshow(np.imag(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[1].set_title('Im{M}', **_title_kw)
        axes[1].set_xlabel(xlabel, **_label_kw)
        fig.colorbar(im1, ax=axes[1])

        if title:
            fig.suptitle(title, **_title_kw)

        for ax in axes:
            _style_ax(ax)
        plt.tight_layout()

    if saveloc:
        plt.savefig(saveloc,dpi=400,bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return

def normalize_rho(rho):
    return rho/np.sum(np.diag(np.abs(rho)))

def normalize_abs(arr):
    return arr/np.sum(np.abs(arr))

def _gauss1d_centered(y, A, s):
    """Single Gaussian centered at 0: A * exp(-0.5 * (y/s)^2)."""
    return A * np.exp(-0.5 * (y / s) ** 2)

def fit_single_gaussian_centered_1d(y_vals, z_vals, n_spike_buffer=1, n_fitting_buffer=3, width_bounds=(0.00, 2.0), center_index=None):
    """Fit a single Gaussian per column, centered at the middle row (fixed center).

    - Recenter y to zero so the Gaussian center is at 0 by construction.
    - Omit n_center_rows around y-center during fitting to avoid the spike.
    - Fit parameters: amplitude A >= 0, width s in [width_bounds[0], width_bounds[1]] * span_y.

    Returns the fitted curve evaluated on the full y grid.

    BOUNDS PERMANENTLY TURNED OFF!!

    """

    ny = y_vals.size

    # Recenter y so 0 is exactly at the Dirac-spike row
    y0 = y_vals[center_index]
    yc = y_vals - y0
    span_y = np.ptp(yc)

    # Leave only two bands around center for fitting
    nlo = center_index - n_spike_buffer - n_fitting_buffer
    nlo1 = center_index - n_spike_buffer - 1
    nhi = center_index + n_spike_buffer + 1
    nhi1 = center_index + n_spike_buffer + n_fitting_buffer + 1

    mask = np.zeros(ny, dtype=bool)
    mask[nlo:nlo1] = True
    mask[nhi:nhi1] = True

    y_fit = yc[mask]
    z_fit = z_vals[mask]

    # # Further threshold-mask low values in z_fit
    # # Compute max on finite values
    # z_max = np.max(z_fit)
    # thr = 0.6 * z_max
    # keep_high = z_fit >= thr
    # frac_kept = float(np.count_nonzero(keep_high)) / float(z_fit.size)
    # # If more than 50% remain after thresholding, return original data
    # if frac_kept > 0.5:
    #     return z_vals
    # # Apply masking; if too few points remain, return original data
    # y_fit = y_fit[keep_high]
    # z_fit = z_fit[keep_high]
    # if z_fit.size < 3:
    #     return z_vals

    # Initial guesses and bounds
    min_s = float(width_bounds[0]) * span_y
    max_s = float(width_bounds[1]) * span_y
    A0 = float(np.median(np.abs(z_fit)))
    s0 = 0.1 * span_y

    try:
        popt, _ = curve_fit(
            _gauss1d_centered,
            y_fit,
            z_fit,
            p0=[A0, s0]#,
            # bounds=([0.0, min_s], [np.inf, max_s]),
            # bounds=([0.0, 0.0], [np.inf, np.inf]),
            # maxfev=20000,
        )
        fit_full = _gauss1d_centered(yc, *popt)
    except:
        fit_full = np.zeros_like(yc)

    return fit_full

###
### Construct interpolated spectra
###


def _sum_n_gauss1d(y, *theta):
    """Sum of n 1D Gaussians (vectorized).

    Parameters are packed as [A1, mu1, s1, A2, mu2, s2, ..., An, mun, sn].
    Returns A1/(2*s1)*exp(-0.5*((y-mu1)/s1)^2) + ...
    """
    y = np.asarray(y, float)
    theta = np.asarray(theta, float)
    if theta.size % 3 != 0:
        raise ValueError("theta length must be a multiple of 3: [A1, mu1, s1, ..., An, mun, sn]")
    n = theta.size // 3
    # Reshape for broadcasting: (n, 3) params vs (N,) y
    p = theta.reshape(n, 3)
    A  = p[:, 0]   # (n,)
    mu = p[:, 1]   # (n,)
    s  = p[:, 2]   # (n,)
    # Mask out zero/negative widths
    valid = s > 0
    if not np.any(valid):
        return np.zeros_like(y)
    A  = A[valid, np.newaxis]    # (m, 1)
    mu = mu[valid, np.newaxis]   # (m, 1)
    s  = s[valid, np.newaxis]    # (m, 1)
    y_row = y[np.newaxis, :]     # (1, N)
    return np.sum(A / (2 * s) * np.exp(-0.5 * ((y_row - mu) / s) ** 2), axis=0)

def fit_n_gaussians_1d(
    y_vals,
    z_vals,
    n,
    use_weights=False,
    width_frac_bounds=(0.01, 0.6),   # min/max sigma as fraction of span_y
    amp_min=0.0,                     # amplitude lower bound
    verbose=True,                    # print status to terminal
    prune_insignificant=True         # collapse negligible components before refinement
):
    """
    Fit a sum of n 1D Gaussians to a 1D signal.

    Strategy:
      1. Multi-scale peak detection (light→heavy smoothing).
      2. Greedy sequential fit: fit one Gaussian at a time to the residual
         so each component locks onto a single feature.
      3. Joint refinement of all parameters together.
      4. Falls back to differential_evolution if local optimisation fails.

    Returns
    -------
    fit_full : ndarray   – model evaluated on y_vals
    popt     : ndarray   – fitted [A1, mu1, s1, ..., An, mun, sn]
    """
    import time as _time
    _t_start = _time.perf_counter()

    def _log(msg):
        if verbose:
            elapsed = _time.perf_counter() - _t_start
            print(f"  [fit_{n}G {elapsed:6.3f}s] {msg}")

    _log(f"starting: {y_vals.size} pts, range [{y_vals[0]:.3g}, {y_vals[-1]:.3g}]")

    y_vals = np.asarray(y_vals, float)
    z_vals = np.asarray(z_vals, float)

    N = y_vals.size
    if N < 3:
        raise ValueError("y_vals must have at least 3 points")

    y_min, y_max = float(y_vals[0]), float(y_vals[-1])
    span_y = float(y_max - y_min)
    if span_y <= 0:
        raise ValueError("y_vals must be strictly increasing")
    dy = span_y / (N - 1)
    n = int(n)

    # ── bounds per component ────────────────────────────────────────
    min_s = max(width_frac_bounds[0] * span_y, 1e-12)
    max_s = max(width_frac_bounds[1] * span_y, min_s * 2)

    lower = np.tile([float(amp_min), y_min, min_s], n)
    upper = np.tile([np.inf,         y_max, max_s], n)

    # ── weights ─────────────────────────────────────────────────────
    sigma_w = None
    if use_weights:
        eps = 1e-12
        sigma_w = np.sqrt(np.abs(z_vals) + eps)
        sigma_w[sigma_w < 1e-12] = 1e-12

    # ── single-component model for greedy stage ─────────────────────
    def _gauss1(y, A, mu, s):
        return A / (2 * s) * np.exp(-0.5 * ((y - mu) / s) ** 2)

    # ── multi-scale peak detection ──────────────────────────────────
    z_pos = np.maximum(z_vals, 0.0)
    z_max = float(z_pos.max()) if z_pos.max() > 0 else 1.0

    prom_threshold = 0.01 * z_max
    min_dist = max(1, N // (5 * max(n, 2)))

    smooth_candidates = sorted(set([
        max(1, N // 200),
        max(1, N // 80),
        max(2, N // 40),
    ]))

    best_peak_idx = np.array([], dtype=int)
    best_widths = np.array([])

    for sm_sigma in smooth_candidates:
        z_smooth = gaussian_filter(z_pos, sigma=sm_sigma)
        trial_idx, trial_props = find_peaks(
            z_smooth, prominence=prom_threshold, distance=min_dist,
        )
        if trial_idx.size > 0:
            trial_widths, _, _, _ = peak_widths(
                z_smooth, trial_idx, rel_height=0.5)
            proms = trial_props['prominences']
            order = np.argsort(proms)[::-1]
            trial_idx = trial_idx[order]
            trial_widths = trial_widths[order]
            if trial_idx.size > best_peak_idx.size:
                best_peak_idx = trial_idx
                best_widths = trial_widths
            if best_peak_idx.size >= n:
                break

    if best_peak_idx.size == 0:
        best_peak_idx = np.array([np.argmax(z_pos)])
        best_widths = np.array([N / 4.0])

    _log(f"peak detection: {best_peak_idx.size} peaks found "
         f"at y=[{', '.join(f'{y_vals[i]:.3g}' for i in best_peak_idx[:6])}]"
         + (" ..." if best_peak_idx.size > 6 else ""))

    # ── greedy sequential fit ───────────────────────────────────────
    # Fit one Gaussian at a time to the residual; this prevents the
    # optimizer from merging nearby peaks into a single wide component.

    fitted_params = []          # list of (A, mu, s) tuples
    residual = z_pos.copy()
    n_detected = best_peak_idx.size

    for k in range(n):
        # Choose initial guess for this component
        if k < n_detected:
            idx = int(best_peak_idx[k])
            s_init = np.clip(best_widths[k] * dy / 2.35, min_s, max_s)
            A_init = max(residual[idx] * 2 * s_init, 1e-6)
            mu_init = float(y_vals[idx])
        else:
            # Pick the argmax of the current residual
            res_pos = np.maximum(residual, 0.0)
            idx = int(np.argmax(res_pos))
            mu_init = float(y_vals[idx])
            s_init = np.clip(0.05 * span_y, min_s, max_s)
            A_init = max(res_pos[idx] * 2 * s_init, 1e-6)

        p0_k = [A_init, mu_init, s_init]
        lo_k = [float(amp_min), y_min, min_s]
        hi_k = [np.inf,         y_max, max_s]

        try:
            popt_k, _ = curve_fit(
                _gauss1, y_vals, np.maximum(residual, 0.0),
                p0=p0_k, bounds=(lo_k, hi_k),
                maxfev=5000, method='trf',
            )
            status = "ok"
        except Exception as e:
            popt_k = np.array(p0_k)
            status = f"fallback ({type(e).__name__})"

        fitted_params.extend(popt_k.tolist())
        # Subtract this component from the residual
        residual = residual - _gauss1(y_vals, *popt_k)
        _log(f"greedy {k+1}/{n}: mu={popt_k[1]:.4g}  s={popt_k[2]:.4g}  A={popt_k[0]:.4g}  [{status}]")

    theta0 = np.asarray(fitted_params, float)
    theta0 = np.clip(theta0, lower + 1e-14, upper - 1e-14)

    # ── prune negligible components ─────────────────────────────────
    # After greedy fitting, some components (especially beyond the number
    # of detected peaks) capture almost nothing.  Collapse them to tiny
    # amplitudes so curve_fit effectively ignores them, reducing the
    # effective dimensionality of the problem.
    if not prune_insignificant:
        # Force only the top 2 components to survive; zero out the rest
        amps_greedy = theta0[0::3].copy()
        if n > 2:
            top2 = np.argsort(amps_greedy)[-2:]
            for k in range(n):
                if k not in top2:
                    theta0[3*k + 0] = lower[3*k + 0] + 1e-14  # A → ~0
            _log(f"forced 2-Gaussian mode: kept components {sorted(top2.tolist())}")

    greedy_rss = float(np.sum((_sum_n_gauss1d(y_vals, *theta0) - z_vals) ** 2))
    _log(f"greedy done — RSS={greedy_rss:.4g}")

    # ── joint refinement ────────────────────────────────────────────
    # Use relaxed tolerances + limited maxfev to keep wall-time bounded.
    def _try_curvefit(p0, maxfev=5000):
        popt, _ = curve_fit(
            _sum_n_gauss1d, y_vals, z_vals,
            p0=p0, bounds=(lower, upper),
            sigma=sigma_w, absolute_sigma=False,
            maxfev=maxfev, method='trf',
            ftol=1e-6, xtol=1e-6, gtol=1e-6,
        )
        fit = _sum_n_gauss1d(y_vals, *popt)
        if np.all(np.isfinite(fit)) and np.any(fit > 0):
            return fit, popt
        raise RuntimeError("bad fit")

    # Measure fit quality: relative sum-of-squares
    def _rss(params):
        fit = _sum_n_gauss1d(y_vals, *params)
        return float(np.sum((fit - z_vals) ** 2))

    best_result = None
    best_rss = np.inf

    # Try joint refinement from greedy init
    try:
        fit, popt = _try_curvefit(theta0)
        rss = _rss(popt)
        if rss < best_rss:
            best_rss = rss
            best_result = (fit, popt)
        _log(f"joint refine (greedy init): RSS={rss:.4g}")
    except Exception as e:
        _log(f"joint refine (greedy init): failed ({type(e).__name__})")

    # Jittered restarts — skip if greedy already gave a good fit,
    # and enforce a wall-time budget.
    greedy_rel_rmse = np.sqrt(greedy_rss / N) / z_max if z_max > 0 else 0.0
    need_jitters = best_result is None or greedy_rel_rmse > 0.02
    n_jitters = 4 if need_jitters else 1
    time_budget = 30.0  # seconds max for jitter phase

    if not need_jitters:
        _log("greedy init good enough — running 1 jitter for safety")

    rng = np.random.default_rng(42)
    for ji in range(n_jitters):
        if _time.perf_counter() - _t_start > time_budget:
            _log(f"jitter {ji+1}/{n_jitters}: skipped (time budget)")
            break
        jitter = theta0.copy()
        jitter[1::3] += rng.uniform(-0.05, 0.05, n) * span_y
        jitter[2::3] *= rng.uniform(0.5, 2.0, n)
        jitter = np.clip(jitter, lower + 1e-14, upper - 1e-14)
        try:
            fit, popt = _try_curvefit(jitter)
            rss = _rss(popt)
            improved = rss < best_rss
            if improved:
                best_rss = rss
                best_result = (fit, popt)
            _log(f"jitter {ji+1}/{n_jitters}: RSS={rss:.4g}{' *' if improved else ''}")
        except Exception:
            _log(f"jitter {ji+1}/{n_jitters}: failed")
            continue

    if best_result is not None:
        elapsed = _time.perf_counter() - _t_start
        rel_rmse = np.sqrt(best_rss / N) / z_max if z_max > 0 else 0.0
        n_sig = int(np.sum(best_result[1][0::3] > 0.01 * best_result[1][0::3].max()))
        _log(f"done — rel_RMSE={rel_rmse:.4f}, {n_sig} significant components, {elapsed:.3f}s total")
        return best_result

    # ── differential evolution fallback ─────────────────────────────
    _log("local fits failed — trying differential evolution (global)...")
    try:
        de_bounds = list(zip(lower, upper))
        z_scale = float(np.max(np.abs(z_vals))) if np.any(z_vals) else 1.0
        for i in range(n):
            if not np.isfinite(de_bounds[3*i][1]):
                de_bounds[3*i] = (de_bounds[3*i][0], 10.0 * z_scale * max_s)

        result = differential_evolution(
            _rss, bounds=de_bounds,
            seed=0, maxiter=600, tol=1e-8,
            x0=theta0, polish=True,
        )
        popt = result.x
        fit_full = _sum_n_gauss1d(y_vals, *popt)
        if np.all(np.isfinite(fit_full)) and np.any(fit_full > 0):
            elapsed = _time.perf_counter() - _t_start
            rel_rmse = np.sqrt(_rss(popt) / N) / z_max if z_max > 0 else 0.0
            _log(f"DE converged — RSS={result.fun:.4g}, rel_RMSE={rel_rmse:.4f}, {elapsed:.3f}s total")
            return fit_full, popt
        _log("DE result invalid (non-finite or all-zero)")
    except Exception as e:
        _log(f"DE failed ({type(e).__name__}: {e})")

    # ── ultimate fallback ───────────────────────────────────────────
    elapsed = _time.perf_counter() - _t_start
    _log(f"WARNING: all strategies failed — returning greedy init as-is ({elapsed:.3f}s total)")
    popt = theta0
    fit_full = _sum_n_gauss1d(y_vals, *popt)
    return fit_full, popt

def spectrum_fun(A,om0,s,om):
    retval = A/(2*s)*np.exp(-(om-om0)**2/(2*s**2))
    return retval  # Positive-freq part

def sp_tot(gausses,om):
    retval = np.zeros_like(om)
    for gauss in gausses:
        _,a0,om0,s0,_ = gauss
        retval += spectrum_fun(a0,om0,s0,om)
    return retval

def extract_midslice(sig_full,slice_fracts,sliced_range,e_slice_fracts=None,e_sliced_range=None):
    if e_slice_fracts is None:
        n_t, n_e = sig_full.shape
        mid_emrange_lo, mid_emrange_hi = slice_fracts

        i0 = floor(n_t*mid_emrange_lo)
        i1 = floor(n_t*mid_emrange_hi)+1

        axis_mid = sliced_range[i0:i1]
        sig_mid = sig_full[i0:i1,:]

        return sig_mid, axis_mid, i0, i1
    else:
        n_t, n_e = sig_full.shape
        mid_emrange_lo, mid_emrange_hi = slice_fracts
        mid_erange_lo, mid_erange_hi = e_slice_fracts

        i0 = floor(n_t*mid_emrange_lo)
        i1 = floor(n_t*mid_emrange_hi)+1

        e_i0 = floor(n_e*mid_erange_lo)
        e_i1 = floor(n_e*mid_erange_hi)+1

        axis_mid = sliced_range[i0:i1]
        e_axis_mid = e_sliced_range[e_i0:e_i1]
        sig_mid = sig_full[i0:i1, e_i0:e_i1]

        return sig_mid, axis_mid, i0, i1, e_axis_mid, e_i0, e_i1

def detrend_spike(sig_mid,row_axis,n_spike_buffer,n_fitting_buffer,N_E=None,above_thresh_mask=False,plot=False):

    abs_mid = np.abs(sig_mid)
    ny, nx = abs_mid.shape
    spike_row_mid = int(np.argmax(np.sum(abs_mid, axis=1)))

    # Fit per column outside the central rows and evaluate on full grid
    fit_cols = np.zeros_like(abs_mid)
    for j in range(nx):
        fit_cols[:, j] = fit_single_gaussian_centered_1d(row_axis, abs_mid[:, j], n_spike_buffer=n_spike_buffer, n_fitting_buffer=n_fitting_buffer, center_index=spike_row_mid)

    # Compute spike as residual but restrict it to central rows only
    spike_mask = np.zeros_like(abs_mid, dtype=bool)
    spike_mask[spike_row_mid-n_spike_buffer:spike_row_mid+n_spike_buffer+1, :] = True

    residual = abs_mid - fit_cols
    residual_pos = np.maximum(residual, 0.0)

    if above_thresh_mask:
        # Build threshold from the row with the maximum average residual and apply globally
        threshold_frac = 0.02  # fraction of strongest-row average to keep
        if ny > 0:
            row_avgs = np.mean(residual_pos, axis=1)
            base_mean = float(row_avgs[np.argmax(row_avgs)])
        else:
            base_mean = 0.0
        thr = threshold_frac * base_mean
        above_thr_mask = residual_pos >= thr

        apply_mask = spike_mask & above_thr_mask
    else:
        apply_mask = spike_mask

    if plot and N_E is not None:
        i_cross = floor(N_E*0.53)

        for v in np.linspace(0,0.99,20):
            i_cross = floor(N_E*v)

            plt.plot(row_axis,abs_mid[:,i_cross],linewidth=2.5,label='Spectrum section')
            plt.plot(row_axis,fit_cols[:,i_cross],label='Fitted baseline')
            # Mark the four fitting-region edges with vertical dotted lines
            c_idx = spike_row_mid
            nlo = c_idx - n_spike_buffer - n_fitting_buffer - 1
            nlo1 = c_idx - n_spike_buffer - 1
            nhi = c_idx + n_spike_buffer + 1
            nhi1 = c_idx + n_spike_buffer + n_fitting_buffer + 1

            plt.axvline(x=row_axis[nlo], color='k', linestyle=':', linewidth=1.2, alpha=0.9, label='Fit data boundaries')
            for idx in (nlo1, nhi, nhi1):
                plt.axvline(x=row_axis[idx], color='k', linestyle=':', linewidth=1.2, alpha=0.9)
            plt.title('Section through signal spectrum')
            plt.xlabel('Indirect energy [eV]')
            plt.ylabel('Amplitude [arb. u.]')
            plt.legend()
            plt.savefig('section.png')
            plt.show()
    
    
    # Spike-only map: positive residuals within central rows and above threshold
    # spike_only = np.where(apply_mask, residual_pos, 0.0)
    spike_only = np.where(apply_mask, residual, 0.0)

    # Baseline uses the fitted 1D Gaussian inside spike areas; elsewhere keep original magnitude
    baseline_mag = abs_mid.copy()
    baseline_mag[apply_mask] = fit_cols[apply_mask]

    # Reconstruct complex result keeping phase but with new baseline magnitude
    eps = 1e-12
    scale = np.divide(np.clip(baseline_mag, 0.0, None), abs_mid + eps)
    scale = np.clip(scale,0,1.05)
    
    amplit_tot_FT_mid_detrended = sig_mid * scale

    return amplit_tot_FT_mid_detrended, spike_only, spike_row_mid

def synth_baseline_n(sp_pr,sp_x,om_pr,om_x,T,mode='same'):
    d_om = om_pr[1]-om_pr[0]

    phase_pr = np.exp(1j*om_pr*T)
    phase_x = np.exp(1j*0*T)

    f1 = sp_pr*phase_pr
    f2 = -sp_x*phase_x/om_x
    conv1 = fftconvolve(f1,f2,mode=mode,axes=1)*d_om

    f1 = sp_x*phase_x
    f2 = -sp_pr*phase_pr/om_pr
    conv2 = fftconvolve(f1,f2,mode=mode,axes=1)*d_om

    return conv1 + conv2

def synth_baseline(sp_pr,sp_x,om_pr,om_x,T,mode='same'):
    d_om = om_x[1]-om_x[0]

    phase0 = np.exp(1j*0*T)
    phase1 = np.exp(1j*om_pr*T)
    phase2 = np.exp(-1j*om_x*T)

    f1 = sp_pr * phase0
    f2 = -sp_x/om_x * phase2
    conv1 = fftconvolve(f1,f2,mode=mode,axes=1)*d_om

    f1 = -sp_pr/om_pr * phase1
    f2 = sp_x * phase0
    conv2 = fftconvolve(f1,f2,mode=mode,axes=1)*d_om

    # return conv1 + conv2
    return conv2

def synth_baseline_hermit(input,sp_x,om_pr,om_x,T):
    d_om = om_x[1]-om_x[0]

    phase0 = np.exp(1j*0*T)
    phase1 = np.exp(1j*om_pr*T)
    phase2 = np.exp(-1j*om_x*T)

    f1 = input
    f2 = -sp_x/om_x * np.conjugate(phase2)
    conv1 = fftconvolve(f1,f2[:,::-1],mode='same',axes=1)*d_om

    f1 = input
    f2 = sp_x * phase0
    conv2 = fftconvolve(f1,f2[:,::-1],mode='same',axes=1)*d_om
    conv2 = -conv2/om_pr * np.conjugate(phase1)

    # return np.sum(conv1 + conv2, axis=0)
    return np.sum(conv2, axis=0)

def plotc(ar):
    plt.plot(np.abs(ar))
    plt.plot(np.real(ar),linewidth=0.8)
    plt.plot(np.imag(ar),linewidth=0.8)
    plt.show()

def nfit_params_to_probes(params, T):
    probes = []
    for k in range(len(params)//3):
        probes.append((T,params[3*k],params[3*k+1],params[3*k+2],0))
    return tuple(probes)

def reconstruct_WirtFlow(sig_measrd,sp_probe,sp_xuv,om_probe,om_xuv,T,b_est,
                         n_power_iter=50,n_main_iter=10000,ifplot=50,naive_init=None,
                         median_regval=2,lastmax_margin=200,ifwait=True,eps=1e-8,alph=0,nt=0):

    om_probe_reg = regularize_omega(om_probe)
    om_xuv_reg = regularize_omega(om_xuv)

    n_t, n_e = sig_measrd.shape

    if naive_init is None:
        w = np.ones((n_e,)).astype(complex) / np.sqrt(n_e)
        poisson_wghts = np.copy(sig_measrd)
        poisson_wghts[sig_measrd < median_regval * np.median(sig_measrd)] = 0

        m = np.size(poisson_wghts)

        for i_iter in range(n_power_iter):
            w = 1/m * synth_baseline_hermit(poisson_wghts * synth_baseline(w,sp_xuv,om_probe_reg,om_xuv_reg,T),sp_xuv,om_probe_reg,om_xuv_reg,T)
            lambda_val0 = np.sqrt(np.sum(np.abs(w)**2))
            w = w / lambda_val0

        lambda_bsln = synth_baseline(w,sp_xuv,om_probe_reg,om_xuv_reg,T)
        alpha = np.sum( np.sqrt((poisson_wghts-b_est)/alph) * np.abs(lambda_bsln) ) / np.sum( np.abs(lambda_bsln)**2 )

        w = alpha*w
    else:
        w = naive_init

    plt.plot(np.real(w),linewidth=0.7)
    plt.plot(np.imag(w),linewidth=0.7)
    plt.plot(np.abs(sp_probe))
    plt.savefig('initial_spectrum_estimate.png')
    plt.close()

    normsq_z0 = np.sum( np.abs(w)**2 )
    z = np.copy(w)

    ers = []
    mus = []
    diffs = []
    coarse_bin = 10
    gif_frame_idx = 0

    # Clear WF_gif folder before saving new frames
    for _old in glob.glob('WF_gif/*'):
        os.remove(_old)

    for i_iter in range(n_main_iter):

        ers.append( np.sum(np.abs( normalize_abs(np.abs(z)) - normalize_abs(np.abs(sp_probe)) )**2)/np.sum(normalize_abs(np.abs(sp_probe))**2) )

        sbforward = synth_baseline(z,sp_xuv,om_probe_reg,om_xuv_reg,T)


        # sbhermit_arg = (np.abs(sbforward)**2 - sig_measrd) * sbforward  GAUSSIAN WF
        lambda_arr = np.abs(sbforward)**2 + b_est
        sbhermit_arg = (1 - sig_measrd/lambda_arr) * sbforward # POISSON WF (??)

        grad = synth_baseline_hermit(sbhermit_arg,sp_xuv,om_probe_reg,om_xuv_reg,T)

        weight_vec = sig_measrd/(lambda_arr)**2
        grad_forward = synth_baseline(grad,sp_xuv,om_probe_reg,om_xuv_reg,T)
        denom = np.sum(np.conjugate(grad_forward)*weight_vec*grad_forward)
        mu_step = np.sum(np.abs(grad)**2)/denom

        z = z - mu_step/normsq_z0 * grad

        mus.append(mu_step)
        diffs.append(np.sum(np.abs(mu_step/normsq_z0 * grad)**2)/np.sum(np.abs(z))**2)

        print(i_iter)

        if i_iter%int(ifplot)==ifplot-1 and bool(ifplot):
            # fig, axes = plt.subplots(2, 1, figsize=(8, 6))
            
            fig1, axes1 = plt.subplots(3, 1, figsize=(8, 10))
            axes1[2].plot(diffs)
            axes1[2].set_yscale('log')

            # axes1[0].plot(np.real(z), label='Re(z)')
            # axes1[0].plot(np.imag(z), label='Im(z)')
            # axes1[0].plot(np.real(sp_probe), label='Re(sp_probe)')
            # axes1[0].plot(np.imag(sp_probe), label='Im(sp_probe)')

            axes1[0].plot(normalize_abs(np.abs(z)), label='Abs(z)')
            axes1[0].plot(normalize_abs(np.abs(sp_probe)), label='Abs(sp_probe)')
            # axes1[0].plot(np.angle(z), label='Abs(z)')
            # axes1[0].plot(np.angle(sp_probe), label='Abs(sp_probe)')

            axes1[0].set_title('Current spectrum estimate')
            axes1[0].legend()

            axes1[1].plot(ers, label='Relative MSE')
            axes1[1].set_yscale('log')
            axes1[1].set_title('Error history')
            axes1[1].set_xlabel('Recorded iteration')
            axes1[1].legend()
            if ers:
                last_val = ers[-1]
                axes1[1].text(
                    0.02, 0.1,
                    f'Iter {i_iter}: {last_val:.5f}',
                    transform=axes1[1].transAxes,
                    fontsize=10,
                    color='white',
                    weight='bold',
                    ha='left',
                    va='top',
                    path_effects=[pe.withStroke(linewidth=1, foreground='black')]
                )

            plt.tight_layout()
            plt.savefig('single_output_temp/WF_diag/spectrum_convergence_iter%.3f_%i.png'%(alph,nt))
            plt.close()

        if i_iter%int(ifplot)==ifplot-1 and bool(ifplot):
            fig = plt.figure(figsize=(12, 8))
            ax_tl = fig.add_subplot(2, 2, 1)
            ax_tr = fig.add_subplot(2, 2, 2)
            ax_bot = fig.add_subplot(2, 1, 2)

            # Top-left: measured signal (2D image)
            ax_tl.imshow(np.abs(sig_measrd), aspect='auto', origin='lower', cmap='plasma')
            ax_tl.set_title('Measured signal',fontsize=10, fontweight='semibold')
            ax_tl.set_xlabel('Energy index')
            ax_tl.set_ylabel('Time index')

            # Top-right: current sbforward (2D image)
            ax_tr.imshow(np.abs(sbforward)**2 + b_est, aspect='auto', origin='lower', cmap='plasma')
            ax_tr.set_title('Current forward model',fontsize=10, fontweight='semibold')
            ax_tr.set_xlabel('Energy index')
            ax_tr.set_ylabel('Time index')

            # Bottom: reconstructed vs true spectrum (abs + phase)
            # Cut omega to [om_lo, om_hi] and convert to wavelength (nm)
            om_lo = 1.5   # lower omega cutoff [rad/fs]
            om_hi = 4   # upper omega cutoff [rad/fs]
            cut_mask = (om_probe >= om_lo) & (om_probe <= om_hi)
            c_nm_fs = 299.792458  # speed of light in nm/fs
            lambda_nm = 2 * np.pi * c_nm_fs / om_probe[cut_mask]

            # Align global phase of z to sp_probe: find phi that minimizes ||z*e^{i*phi} - sp_probe||
            global_phase = np.angle(np.sum(np.conj(z) * sp_probe))
            z_aligned = z * np.exp(1j * global_phase)

            abs_z = normalize_abs(np.abs(z_aligned[cut_mask]))
            abs_sp = normalize_abs(np.abs(sp_probe[cut_mask]))
            ax_bot.plot(lambda_nm, abs_z, label='Reconstructed (abs)', linewidth = 1.2)
            ax_bot.plot(lambda_nm, abs_sp, label='Reference (abs)', alpha=0.7, linewidth = 1.8)

            # Phase masked where magnitude < 5% of peak
            phase_z = np.angle(z_aligned[cut_mask]).copy()
            phase_sp = np.angle(sp_probe[cut_mask]).copy()
            thresh_z = 0.05 * np.max(np.abs(z_aligned))
            thresh_sp = 0.05 * np.max(np.abs(sp_probe))
            phase_z[np.abs(z[cut_mask]) < thresh_z] = np.nan
            phase_sp[np.abs(sp_probe[cut_mask]) < thresh_sp] = np.nan

            ax_phase = ax_bot.twinx()
            ax_phase.plot(lambda_nm, phase_z+np.pi, '--', label='Reconstructed (phase)', color='C0', alpha=0.6)
            ax_phase.plot(lambda_nm, phase_sp+np.pi, '--', label='Reference (phase)', color='C1', alpha=0.6)
            ax_phase.set_ylabel('Phase [rad]')
            ax_phase.set_ylim(-0.2,2*np.pi)

            # Combine legends from both axes
            lines1, labels1 = ax_bot.get_legend_handles_labels()
            lines2, labels2 = ax_phase.get_legend_handles_labels()
            ax_bot.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

            ax_bot.set_xlabel('Wavelength [nm]')
            ax_bot.set_title('Spectrum estimate',fontsize=10, fontweight='semibold')
            ax_bot.text(
                0.02, 0.06,
                f'Iter {i_iter+1}',
                transform=ax_bot.transAxes,
                fontsize=14,
                color='white',
                weight='bold',
                ha='left',
                va='bottom',
                path_effects=[pe.withStroke(linewidth=2, foreground='black')]
            )

            plt.tight_layout()
            # plt.savefig('single_output_temp/WF_diag/spectrum_convergence_iter%.3f_%i.png'%(alph,nt))
            plt.savefig('WF_gif/frame_%04d.png' % gif_frame_idx, dpi=100)
            plt.close()
            gif_frame_idx += 1

            if diffs[-1] < eps:
                break

        # if i_iter%coarse_bin == coarse_bin-1:
            # avgs = np.mean(np.abs(mus).reshape(-1, coarse_bin), axis=1)
            # if i_iter - np.argmax(avgs)*10 > lastmax_margin and avgs[-1] < avgs[-2] and (not ifwait or avgs[-1] > avgs[0]) :
            #     break


    # Assemble GIF from saved frames
    frame_files = sorted(glob.glob('WF_gif/frame_*.png'))
    if frame_files:
        frames = [Image.open(f) for f in frame_files]
        frames[0].save(
            'WF_gif/convergence.gif',
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0
        )
        for f in frames:
            f.close()

    return z, ers[-1]

def clip_amplitude(sig,c0,c1):
    amp = np.abs(sig)
    phase = np.angle(sig)
    return np.clip(amp,c0,c1)*np.exp(1j*phase)

def threshold_noise(M_obs, mean_noise_power):
    n_t, n_e = M_obs.shape
    # 1. Derive sigma from Mean Noise Power (2s^2)
    # sigma = sqrt( Power / 2 )
    sigma = np.sqrt(mean_noise_power / 2.0)
    
    # 2. Calculate the Noise Floor (The Expected Mean of Pure Noise)
    # Floor = sigma * sqrt(pi/2)
    noise_floor = sigma * np.sqrt(np.pi / 2.0)

    A_est = np.zeros_like(M_obs)
    # 3. Iterate through data
    for j in range(n_e):
        for i, m in enumerate(M_obs[:,j]):
            # Case 1: Signal is below or equal to the noise floor
            if m <= noise_floor[j]:
                A_est[i,j] = 0.0
            else:
                A_est[i,j] = m
    return A_est

def koay_basser_correction(M_obs, mean_noise_power,lambda_thresh=1):
    """
    Removes Rician bias using the Koay-Basser inversion method.
    
    Parameters:
    -----------
    M_obs : float or array-like
        The observed magnitude (SPD amplitude).
    mean_noise_power : float
        The known bias in the power domain (2*sigma^2).
        
    Returns:
    --------
    A_est : float or array-like
        The corrected signal amplitude.
    """
    # 1. Derive sigma from Mean Noise Power (2s^2)
    # sigma = sqrt( Power / 2 )
    sigma = np.sqrt(mean_noise_power / 2.0)
    
    # 2. Calculate the Noise Floor (The Expected Mean of Pure Noise)
    # Floor = sigma * sqrt(pi/2)
    noise_floor = sigma * np.sqrt(np.pi / 2.0)
    
    # Handle array inputs
    M_obs = np.atleast_1d(M_obs)
    A_est = np.zeros_like(M_obs)
    
    # Define the objective function for root finding
    # We solve for SNR_A (A/sigma) to keep numbers well-scaled
    def rician_mean_eq(snr_a, snr_m_target):
        # y = (A^2) / (4*sigma^2) = snr_a^2 / 4
        y = (snr_a**2) / 4.0
        
        # Theoretical Mean / sigma
        # Using exponentially scaled Bessel functions (i0e, i1e) to prevent overflow
        # The exp(-y) term cancels out with the scaling of i0e and i1e
        f_val = np.sqrt(np.pi / 2.0) * ((1 + 2*y) * i0e(y) + 2*y * i1e(y))
        
        return f_val - snr_m_target

    # 3. Iterate through data
    for i, m in enumerate(M_obs):
        # Case 1: Signal is below or equal to the noise floor
        if m <= lambda_thresh*noise_floor:
            A_est[i] = 0.0
        
        # Case 2: Signal is above noise floor -> Invert the function
        else:
            current_snr_m = m / sigma
            
            # Use Brent's method to find the root.
            # Lower bound 0, Upper bound usually just needs to be high enough.
            # Since M > A usually, current_snr_m is a safe heuristic upper bound 
            # for the search, but we add a buffer for safety.
            try:
                root_snr = brentq(rician_mean_eq, 0, current_snr_m * 2, args=(current_snr_m,))
                A_est[i] = root_snr * sigma
            except ValueError:
                # Fallback if convergence fails (rare)
                A_est[i] = m 

    return A_est if len(A_est) > 1 else A_est[0]

def modulating_function_multi(om_t,ene_Eg,pulse_params_x,probes,big_sigma=0):
    ###
    ### MODULATING FUNCTION NEEDS TO TAKE INTO ACCOUNT FT WINDOW IF USED (not done)

    retval = 0
    A_tot = 0

    for xuv_now in pulse_params_x:
        for probe in probes:

            tau_x,A_x,om0_x,s_x,phi_x = xuv_now

            tau_p,A_p,om0_p,s_p,phi_p = probe
            s_pp = s_p+big_sigma

            # Frequency-domain widths combine as s = sqrt(s_x^2 + s_p^2)
            s = np.sqrt(s_x**2 + s_pp**2)
            # Time-domain width for the cross term (amplitude-level) combines as
            # s_t = sqrt(1/s_x^2 + 1/s_p^2)
            s_t = np.sqrt(1/s_x**2 + 1/s_pp**2)

            delta_p = om0_x + om0_p - ene_Eg/hbar
            # Center of the ω_t Gaussian for the cross term in the amplitude model
            # Derived form: om_t center at -delta_E, where
            # delta_E = om0_x - ene_Eg/hbar - (s_x^2/s^2) * delta_p
            delta_E = om0_x - ene_Eg/hbar - (s_x**2/s**2) * delta_p

            # Use amplitude detuning factor (delta_p/s)^2, and ω_t Gaussian centered at -delta_E
            retval += A_x * A_p/(s_pp) * np.exp(-0.5*((delta_p/s)**2 + (s_t * (om_t + delta_E))**2))

    return retval

def normalize_params(gauss_params, om):
    """
    Normalize a tuple of Gaussian parameters so their sum integrates to 1.
    
    Args:
        gauss_params: tuple of tuples, each (tau, A, om0, s, phi)
        om: array of omega values for integration
        
    Returns:
        tuple of normalized tuples in same format as input
    """
    # Compute total spectrum
    total_spectrum = sp_tot(gauss_params, om)
    
    # Compute integral using trapezoidal rule
    d_om = om[1] - om[0]
    integral = np.sum(total_spectrum) * d_om
    
    if integral == 0.0 or not np.isfinite(integral):
        # Spectrum has no overlap with the grid — return unmodified params
        import warnings
        warnings.warn(
            "normalize_params: integral is zero or non-finite "
            "(Gaussian centers likely outside the provided grid). "
            "Returning un-normalised parameters.",
            RuntimeWarning,
            stacklevel=2,
        )
        return gauss_params

    # Scale all amplitudes by the normalization factor
    norm_factor = 1.0 / integral
    normalized = tuple(
        (tau, A * norm_factor, om0, s, phi)
        for tau, A, om0, s, phi in gauss_params
    )
    
    return normalized

def correcting_function_multi(om_t,ene_Eg,pulse_params_x,probes,dzeta=1e-6,theta=1):
    mod_plus = modulating_function_multi(om_t,ene_Eg,pulse_params_x,probes)
    err_area_plus = np.where(mod_plus<dzeta)
    mod_plus[err_area_plus] = 1
    mod2_plus = modulating_function_multi(om_t,ene_Eg,pulse_params_x,probes,big_sigma=1000)

    correction_plus = om_t*mod2_plus/mod_plus
    correction_plus[correction_plus>theta] = 0
    correction_plus[err_area_plus] = 0

    return correction_plus

def correcting_function_synth(T_range, T_grid,
                               sp_probe, sp_xuv, om_probe, om_xuv,
                               sp_ref=None,
                               p_E=1,
                               dzeta=1e-6, theta=1, use_window=True):
    """
    Compute correction function from the ratio of synthetic signal FTs.

    Instead of relying on a Gaussian decomposition of the probe, this function
    generates clean (noise-free, high-resolution) synthetic signals using
    ``synth_baseline`` with the *actual* probe spectrum and with a *flat*
    (constant-amplitude) probe, FTs both along time, and returns the
    regularised magnitude ratio:

        correction = |FT(sig_flat)| / |FT(sig_actual)|

    The ratio undoes the probe spectral-shape modulation imprinted on the
    measured sideband signal, without assuming any parametric form for the
    probe.

    Parameters
    ----------
    T_range : 1-D array (N_T,)
        Uniformly spaced time sample points.
    T_grid : 2-D array (N_T, N_E_up)
        Time meshgrid matching the spectral arrays (e.g. ``self.T_up``).
    sp_probe : 1-D array (N_E_up,)
        Probe spectrum evaluated on *om_probe*.
    sp_xuv : 1-D array (N_E_up,)
        XUV spectrum evaluated on *om_xuv*.
    om_probe : 1-D array (N_E_up,)
        Probe frequency grid.
    om_xuv : 1-D array (N_E_up,)
        XUV frequency grid.
    sp_ref : 1-D array (N_E_up,), optional
        Reference-arm spectrum on *om_probe*.  When given, the ref
        cross-term (at T = 0) is added to both synthetic signals so that
        probe–ref interference is properly accounted for.
    p_E : int
        Energy down-sampling factor (> 1 when spectra are upsampled).
    dzeta : float
        Where |FT_actual| < dzeta the correction is set to zero.
    theta : float
        Where the correction exceeds theta it is set to zero.
    use_window : bool
        Apply a Hann window in the CFT (should match the data processing).

    Returns
    -------
    correction : 2-D array (N_T, N_E)
        Multiplicative real correction in the (omega_t, E) domain.
    """
    om_probe_reg = regularize_omega(om_probe)
    om_xuv_reg = regularize_omega(om_xuv)

    # --- synthetic signal with actual probe ---
    amplit_actual = synth_baseline(sp_probe, sp_xuv,
                                   om_probe_reg, om_xuv_reg, T_grid)
    if sp_ref is not None:
        amplit_actual += synth_baseline(sp_ref, sp_xuv,
                                        om_probe_reg, om_xuv_reg, 0 * T_grid)
    if p_E > 1:
        amplit_actual = downsample(amplit_actual, p_E)
    sig_actual = np.abs(amplit_actual) ** 2

    # --- synthetic signal with flat (constant) probe ---
    sp_flat = np.ones_like(sp_probe)
    amplit_flat = synth_baseline(sp_flat, sp_xuv,
                                 om_probe_reg, om_xuv_reg, T_grid)
    if sp_ref is not None:
        amplit_flat += synth_baseline(sp_ref, sp_xuv,
                                      om_probe_reg, om_xuv_reg, 0 * T_grid)
    if p_E > 1:
        amplit_flat = downsample(amplit_flat, p_E)
    sig_flat = np.abs(amplit_flat) ** 2

    # --- FT both along the time axis ---
    FT_actual, _, _, _ = CFT(T_range, sig_actual, use_window=use_window)
    FT_flat,   _, _, _ = CFT(T_range, sig_flat,   use_window=use_window)

    # --- regularised ratio ---
    abs_actual = np.abs(FT_actual)
    abs_flat   = np.abs(FT_flat)

    err_area = np.where(abs_actual < dzeta)
    abs_actual[err_area] = 1.0            # safe denominator

    correction = abs_flat / abs_actual
    correction[correction > theta] = 0.0
    correction[err_area] = 0.0

    return correction

def new_Sig_cc_interp(re_spline, im_spline, om_t_vals, Ef_vals, grid=None):
    """
    Evaluate the cubic splines.
    - If om_t_vals and Ef_vals are 1D arrays and grid=True/None: returns a 2D grid evaluation.
    - If om_t_vals and Ef_vals are 2D arrays (meshgrid): evaluates pointwise on that mesh.
    - Otherwise, broadcasting rules are applied and evaluated pointwise.
    """
    om_t_vals = np.asarray(om_t_vals)
    Ef_vals = np.asarray(Ef_vals)

    # Case 1: 1D axes -> grid evaluation
    if om_t_vals.ndim == 1 and Ef_vals.ndim == 1 and (grid is None or grid is True):
        re = re_spline(om_t_vals, Ef_vals, grid=True)
        im = im_spline(om_t_vals, Ef_vals, grid=True)
        return re + 1j * im

    # Case 2: meshgrid or arbitrary broadcastable arrays -> pointwise evaluation
    om_b, ef_b = np.broadcast_arrays(om_t_vals, Ef_vals)
    re = re_spline.ev(om_b.ravel(), ef_b.ravel()).reshape(om_b.shape)
    im = im_spline.ev(om_b.ravel(), ef_b.ravel()).reshape(om_b.shape)
    return re + 1j * im

def resample(spec_corrected,rho_hi,rho_lo,om_ref,E,OM_T,N_NEW):

    rho_valrange = rho_hi-rho_lo

    new_Ef_min = rho_lo + hbar*om_ref
    new_Ef_max = rho_hi + hbar*om_ref

    new_omt_min = om_ref - rho_valrange/hbar
    new_omt_max = om_ref + rho_valrange/hbar

    idx_min = np.argmin(np.abs(new_Ef_min-E[0,:]))
    idx_max = np.argmin(np.abs(new_Ef_max-E[0,:]))

    idy_min = np.argmin(np.abs(new_omt_min-OM_T[:,0]))
    idy_max = np.argmin(np.abs(new_omt_max-OM_T[:,0]))

    extent = [E[0,idx_min], E[0,idx_max], hbar*OM_T[idy_min,0], hbar*OM_T[idy_max,0]]

    small_sig = spec_corrected[idy_min:idy_max,idx_min:idx_max]

    # Use the actual sampled axes from the selected window to avoid half-bin shifts
    Ef_range = E[0, idx_min:idx_max]
    omt_range = OM_T[idy_min:idy_max, 0]

    re_spline = RectBivariateSpline(omt_range*hbar, Ef_range, np.real(small_sig), kx=3, ky=3, s=0)
    im_spline = RectBivariateSpline(omt_range*hbar, Ef_range, np.imag(small_sig), kx=3, ky=3, s=0)

    # Create grid directly in the desired (e1, e2) coordinates
    # Both e1 and e2 should range over [rho_lo, rho_hi] in the OUTPUT
    e1_range = np.linspace(rho_lo, rho_hi, N_NEW)
    e2_range = np.linspace(rho_lo, rho_hi, N_NEW)
    
    E1, E2 = np.meshgrid(e1_range, e2_range, indexing='ij')
    
    # Inverse transformation to get (ℏ·ωt, Ef) for spline evaluation
    # Desired: e1 = Ef - ℏ·ωt, e2 = Ef (in the final rotated frame)
    # But the spline is defined on the ORIGINAL (ℏ·ωt, Ef) coordinates
    # where Ef is shifted by hbar*om_ref relative to the rho scale
    # 
    # The mapping should be:
    # Ef_spline = e2 + hbar*om_ref (shift e2 to spline's Ef domain)
    # ℏ·ωt_spline = Ef_spline - e1 = (e2 + hbar*om_ref) - e1
    EPS1 = E2 - E1 + hbar*om_ref  # This is ℏ·ωt (first argument to spline)
    EPS2 = E2 + hbar*om_ref       # This is Ef (second argument to spline)

    Sig_cc_cubic_mesh = new_Sig_cc_interp(re_spline, im_spline, EPS1, EPS2)

    # Sig_cc_cubic_mesh = 1/2*(Sig_cc_cubic_mesh + np.conjugate(Sig_cc_cubic_mesh.T))

    # Sig_cc_cubic_mesh = Sig_cc_cubic_mesh/np.sum(np.diag(np.abs(Sig_cc_cubic_mesh)))

    return Sig_cc_cubic_mesh, small_sig, extent, [idy_min,idy_max,idx_min,idx_max], E1, E2

def downsample(sig, p):
    n, m = sig.shape
    if m % p != 0:
        raise ValueError("Number of columns must be divisible by p")
    k = m // p
    return sig.reshape(n, k, p).mean(axis=2)

def project_to_density_matrix(M, smooth_sigma=1.5):
    """
    Project a noisy measured matrix onto the set of valid density matrices
    (Hermitian, PSD, trace 1).

    Simple and robust approach for experimental data:

    1.  Hermitize.
    2.  Gaussian-smooth the full matrix (real & imag independently).
        Smoothing *before* eigendecomposition is the key: the eigenvectors
        of a smooth matrix are themselves smooth, so the reconstruction
        ``V diag(λ) V†`` cannot produce row/column line artifacts.
    3.  Eigendecompose, clip negative eigenvalues to zero, normalise to
        trace 1, reconstruct.

    Parameters
    ----------
    M : np.ndarray
        Measured matrix (complex, shape d×d).
    smooth_sigma : float
        Gaussian σ (pixels) for 2-D smoothing of the matrix.  Default 1.5.

    Returns
    -------
    np.ndarray
        Valid density matrix (Hermitian, PSD, trace 1).
    """
    d = M.shape[0]

    # 1. Hermitize
    H = (M + M.conj().T) / 2

    # 2. Gaussian smooth (real & imag independently), then re-Hermitize
    if smooth_sigma > 0:
        H = (gaussian_filter(np.real(H), sigma=smooth_sigma)
             + 1j * gaussian_filter(np.imag(H), sigma=smooth_sigma))
        H = (H + H.conj().T) / 2

    # 3. Eigendecompose, clip negative eigenvalues, normalise trace
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals, 0.0)

    eig_sum = np.sum(eigvals)
    if eig_sum <= 0:
        return np.eye(d, dtype=complex) / d
    eigvals = eigvals / eig_sum

    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    return rho

def fidelity(rho, sigma):
    """
    Compute the Uhlmann fidelity between two density matrices.
    
    Parameters:
        rho (np.ndarray): density matrix
        sigma (np.ndarray): density matrix

    Returns:
        float: fidelity value
    """
    # Safety: ensure Hermitian
    rho = (rho + rho.conj().T) / 2
    sigma = (sigma + sigma.conj().T) / 2

    # Compute root of rho
    sqrt_rho = sqrtm(rho)

    # Compute the intermediate matrix
    inner = sqrt_rho @ sigma @ sqrt_rho

    # Compute the fidelity
    fidelity = np.real((np.trace(sqrtm(inner)))**2)

    return float(fidelity)

def regularize_omega(om):
    """
    Regularize an array of omega values around zero crossing.
    
    For values close to zero, replace with a constant (negative before zero, 
    positive after zero). Values far from zero remain unchanged.
    
    Parameters:
        om : array-like
            Regularly spaced array of floats
            
    Returns:
        array-like
            Regularized omega array
    """
    om = np.asarray(om)
    result = np.copy(om)
    
    # Determine spacing to define "close to zero"
    d_om = np.abs(om[1] - om[0]) if len(om) > 1 else 1.0
    threshold = 10 * d_om  # Regularize within ~4 grid spacings of zero
    
    # Find indices close to zero
    close_to_zero = np.abs(om) <= threshold
    
    if np.any(close_to_zero):
        # Find the zero crossing point
        zero_idx = np.argmin(np.abs(om))
        
        # Regularize: negative constant before zero, positive after
        before_zero = (om < 0) & close_to_zero
        after_zero = (om >= 0) & close_to_zero
        
        result[before_zero] = -threshold
        result[after_zero] = threshold
    
    return result

def pulse_emit(probes):
    """
    Take a tuple of pulse values and flip the omega to negative in all of them.
    
    Parameters:
    -----------
    probes : tuple
        Tuple of pulse tuples, each in format (tau, amplitude, omega, sigma, phi)
    
    Returns:
    --------
    tuple
        Tuple of pulse tuples with omega values flipped to negative
    """
    flipped_probes = []
    for probe in probes:
        tau, amplitude, omega, sigma, phi = probe
        flipped_probe = (tau, amplitude, -omega, sigma, phi)
        flipped_probes.append(flipped_probe)
    
    return tuple(flipped_probes)

    
def conv_bounds(result_lo, result_hi, N, s1_center):
    """Compute signal axis bounds so that fftconvolve(sig_1, sig_2, mode='same')
    corresponds to np.linspace(result_lo, result_hi, N).

    Both signals are sampled with N points at the same spacing
    dx = (result_hi - result_lo) / (N - 1), so each axis spans
    (result_hi - result_lo).  The free parameter `s1_center` sets
    the centre of the first signal's axis; the second signal's
    axis is fully determined by the convolution alignment constraint.

    Parameters
    ----------
    result_lo, result_hi : float
        Desired lower / upper bounds of the convolution result axis.
    N : int
        Number of sample points (same for both signals and the result).
    s1_center : float
        Centre of sig_1's axis.

    Returns
    -------
    s1_lo, s2_lo, s1_hi, s2_hi : float
    """
    L  = result_hi - result_lo          # span of each signal axis
    dx = L / (N - 1)
    # 'same' starts at index (N-1)//2 of the full (2N-1)-point result,
    # so the required sum of the two lower bounds is:
    sum_lo = result_lo - ((N - 1) // 2) * dx

    s1_lo = s1_center - L / 2
    s2_lo = sum_lo - s1_lo
    s1_hi = s1_lo + L
    s2_hi = s2_lo + L
    return s1_lo, s2_lo, s1_hi, s2_hi