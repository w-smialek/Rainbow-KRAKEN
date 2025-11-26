import numpy as np
from numpy import pi
from scipy.special import erf
from scipy.special import wofz
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate
from scipy import integrate
from math import floor, ceil
from scipy.optimize import curve_fit
from scipy.signal import deconvolve
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import savgol_filter
from scipy.signal import fftconvolve
from matplotlib import patheffects as pe

# Reduced Planck constant in eV*fs (approx CODATA): ħ = 6.582119569e-16 eV·s
hbar = 6.582119569e-1

def Amplitude_ij(pulse_params_i,pulse_params_j,ene_Eg):
    tau_i,A_i,om0_i,s_i,phi_i = pulse_params_i
    tau_j,A_j,om0_j,s_j,phi_j = pulse_params_j

    tau = tau_i - tau_j

    s = np.sqrt(s_i**2+s_j**2)
    s_t = np.sqrt(s_i**(-2)+s_j**(-2))
    delta = om0_i + om0_j - ene_Eg/hbar
    wofz_arg = s_t/np.sqrt(2)*((om0_j-s_j**2/s**2*delta-1j*tau/s_t**2) - ene_Eg/hbar)

    prefactor = (-pi*A_i*A_j/(4*s_i*s_j)*np.exp(-1j*phi_i*np.sign(om0_i))*np.exp(-1j*phi_j*np.sign(om0_j))*np.exp(1j*om0_i*tau)
                * np.exp(-1/2*( delta**2/s**2 + tau**2/s_t**2 + 2j*s_i/s_j*delta/s*tau/s_t )))
    return prefactor*wofz(wofz_arg)

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

    windowed = (signal * window) / coherent_gain

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

    OM_T = (energy_axis_shift / hbar)  # back to angular frequency ω
    OM_T = np.tile(OM_T, (NE_local, 1)).T

    return spec_shift, OM_T, energy_axis_shift[0], energy_axis_shift[-1]

def plot_mat(mat,extent=[0,1,0,1],cmap='plasma',mode='phase',show=True,saveloc=None,caption=None,xlabel='x',ylabel='y',title=None):

    if mode == 'abs':
        plt.figure()
        im = plt.imshow(np.abs(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        else:
            plt.title('|M|')
        plt.colorbar(im)
        if caption is not None:
            plt.text(0.02, 0.98, caption, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='left',
                    color='white', weight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    elif mode == 'phase':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        im0 = axes[0].imshow(np.abs(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[0].set_title('|M|')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        fig.colorbar(im0, ax=axes[0])
        if caption is not None:
            axes[0].text(0.02, 0.98, caption, transform=axes[0].transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='left',
                        color='white', weight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        im1 = axes[1].imshow(np.angle(mat), extent=extent, origin='lower', aspect='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('arg(M)')
        axes[1].set_xlabel(xlabel)
        fig.colorbar(im1, ax=axes[1])

        if title:
            fig.suptitle(title)

        plt.tight_layout()

    elif mode == 'reim':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        im0 = axes[0].imshow(np.real(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[0].set_title('Re{M}')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        fig.colorbar(im0, ax=axes[0])
        if caption is not None:
            axes[0].text(0.02, 0.98, caption, transform=axes[0].transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='left',
                        color='white', weight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        im1 = axes[1].imshow(np.imag(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[1].set_title('Im{M}')
        axes[1].set_xlabel(xlabel)
        fig.colorbar(im1, ax=axes[1])

        if title:
            fig.suptitle(title)

        plt.tight_layout()

    if saveloc:
        plt.savefig(saveloc,dpi=400)
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

    # if j%100 == 0:
    #     plt.plot(y_fit,z_fit+0.1)
    #     plt.plot(yc,z_vals)
    #     plt.plot(yc,fit_full)
    #     plt.show()

    return fit_full

###
### Construct interpolated spectra
###


def _sum_n_gauss1d(y, *theta):
    """Sum of n 1D Gaussians.

    Parameters are packed as [A1, mu1, s1, A2, mu2, s2, ..., An, mun, sn].
    Returns A1*exp(-0.5*((y-mu1)/s1)^2) + ...
    """
    y = np.asarray(y)
    theta = np.asarray(theta, float)
    if theta.size % 3 != 0:
        raise ValueError("theta length must be a multiple of 3: [A1, mu1, s1, ..., An, mun, sn]")
    n = theta.size // 3
    out = np.zeros_like(y, dtype=float)
    for k in range(n):
        A = theta[3*k + 0]
        mu = theta[3*k + 1]
        s = theta[3*k + 2]
        if s <= 0:
            continue
        out += A * np.exp(-0.5 * ((y - mu) / s) ** 2)
    return out

def fit_n_gaussians_1d(
    y_vals,
    z_vals,
    n,
    mask=None,
    center_bounds=None,
    width_bounds_frac=(1e-3, 3.0),
    amp_bounds=(0.0, np.inf),
    use_weights=True,
    return_params=False,
    return_components=False,
    resample_y=None
):
    """Fit a non-negative 1D signal as a sum of n Gaussians.

    Args:
        y_vals: 1D coordinate array (monotonic preferred)
        z_vals: 1D data array (ideally non-negative)
        n: number of Gaussian components
        mask: optional boolean mask selecting samples to use in the fit
        center_bounds: tuple (y_min, y_max) for all centers; default is (min(y), max(y))
        width_bounds_frac: (min_frac, max_frac) of span(y) for s bounds
        amp_bounds: (A_min, A_max) bounds for amplitudes (default non-negative)
        use_weights: if True and z_vals >= 0, weight by 1/sqrt(z + eps)
        return_params: if True, also return fitted parameter vector (3n,)
        return_components: if True, also return individual Gaussian components as (n, len(y))

    Returns:
        fit_full: fitted signal on full y grid
        [params]: optional, flat parameter vector [A1, mu1, s1, ..., An, mun, sn]
        [components]: optional, array of shape (n, len(y)) with each Gaussian
    """
    y = np.asarray(y_vals).ravel()
    z = np.asarray(z_vals).ravel()
    if y.size != z.size:
        raise ValueError("y_vals and z_vals must have the same length")
    N = y.size
    if N == 0 or int(n) <= 0:
        return (np.zeros_like(z),) if not return_params and not return_components else (np.zeros_like(z), np.array([]))

    # Select samples
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.size != N:
            raise ValueError("mask must have the same length as y_vals")
        y_fit = y[mask]
        z_fit = z[mask]
    else:
        y_fit = y
        z_fit = z

    # Bounds
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span_y = max(y_max - y_min, 1e-12)
    if center_bounds is None:
        center_bounds = (y_min, y_max)
    min_s = max(float(width_bounds_frac[0]) * span_y, 1e-12)
    max_s = max(float(width_bounds_frac[1]) * span_y, min_s)

    # Initial guesses: centers evenly spaced, amps from local sampling, widths as span fraction
    mu0s = np.linspace(y_min, y_max, int(n) + 2)[1:-1]
    # Sample z at nearest indices for initial amplitudes
    if y_max > y_min:
        A0s = []
        for mu0 in mu0s:
            idx = int(np.clip(np.searchsorted(y, mu0), 0, N-1))
            A0s.append(max(z[idx], 1e-12))
        A0s = np.asarray(A0s)
    else:
        A0s = np.full(int(n), max(np.median(np.abs(z_fit)), 1e-12))
    s0 = 0.1 * span_y if span_y > 0 else 1.0
    theta0 = []
    lb = []
    ub = []
    for k in range(int(n)):
        theta0.extend([float(A0s[k]), float(mu0s[k]), float(s0)])
        lb.extend([float(amp_bounds[0]), float(center_bounds[0]), float(min_s)])
        ub.extend([float(amp_bounds[1]), float(center_bounds[1]), float(max_s)])
    theta0 = np.asarray(theta0)
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    # Weights (optional): Poisson-like -> sigma ~ sqrt(z + eps)
    sigma = None
    if use_weights and np.all(z_fit >= 0):
        eps = 1e-12
        sigma = np.sqrt(z_fit + eps)
        # Avoid zero sigma
        sigma[sigma < 1e-12] = 1e-12

    popt, _ = curve_fit(
        _sum_n_gauss1d,
        y_fit,
        z_fit,
        p0=theta0,
        bounds=(lb, ub),
        sigma=sigma,
        absolute_sigma=False,
        maxfev=200000,
    )

    # Evaluate on full grid
    if resample_y is not None:
        fit_full = _sum_n_gauss1d(resample_y, *popt)
    else:
        fit_full = _sum_n_gauss1d(y, *popt)
    if not return_components and not return_params:
        return fit_full

    out = [fit_full]
    if return_params:
        out.append(popt)
    if return_components:
        comps = np.zeros((int(n), y.size))
        for k in range(int(n)):
            A, mu, s = popt[3*k:3*k+3]
            comps[k] = A * np.exp(-0.5 * ((y - mu) / s) ** 2)
        out.append(comps)
    return tuple(out)


def spectrum_fun(A,om0,s,om):
    retval = A/(2*s)*np.exp(-(om-om0)**2/(2*s**2))
    return retval  # Positive-freq part

def sp_tot(gausses,om):
    retval = np.zeros_like(om)
    for gauss in gausses:
        _,a0,om0,s0,_ = gauss
        retval += spectrum_fun(a0,om0,s0,om)
    return retval

def spectrum_fun_2d(A,om0,s,OM,TAU):
    retval = A/(2*s)*np.exp(-(OM-om0)**2/(2*s**2))*np.exp(1j*OM*TAU)
    return retval  # Positive-freq part

def sp_tot_2d(gausses,OM,TAU):
    retval = np.zeros_like(OM).astype(complex)
    for gauss in gausses:
        _,a0,om0,s0,_ = gauss
        retval += spectrum_fun_2d(a0,om0,s0,OM,TAU)
    return retval

def Amplitude(xuvs,refprobes,E,E_spinorbit=0,order=None):
    n_t, n_e = E.shape
    amplit_tot = np.zeros((n_t,n_e)).astype(complex)

    xuvs_mod = []
    for xuv in xuvs:
        tau_i,A_i,om0_i,s_i,phi_i = xuv
        xuvs_mod.append((tau_i,A_i,om0_i-E_spinorbit/hbar,s_i,phi_i))

    if order == None:
        for xuv in xuvs_mod:
            for rp in refprobes:
                amplit_tot += Amplitude_ij(xuv,rp,E)
                amplit_tot += Amplitude_ij(rp,xuv,E)
    elif order == 1:
        for xuv in xuvs_mod:
            for rp in refprobes:
                amplit_tot += Amplitude_ij(xuv,rp,E)
    else:
        for xuv in xuvs_mod:
            for rp in refprobes:
                amplit_tot += Amplitude_ij(rp,xuv,E)

    return amplit_tot

def extract_midslice(sig_full,slice_fracts,sliced_range):
    n_t, n_e = sig_full.shape
    mid_emrange_lo, mid_emrange_hi = slice_fracts

    i0 = floor(n_t*mid_emrange_lo)
    i1 = floor(n_t*mid_emrange_hi)+1

    axis_mid = sliced_range[i0:i1]
    sig_mid = sig_full[i0:i1,:]

    return sig_mid, axis_mid, i0, i1

def detrend_spike(sig_mid,row_axis,n_spike_buffer,n_fitting_buffer,above_thresh_mask=False,plot=False):

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

    if plot:
        i_cross = floor(N_E*0.53)

        for v in [0.2,0.35,0.45,0.50,0.55,0.60,0.85]:
            i_cross = floor(N_E*v)

            plt.plot(em_axis_mid,abs_mid[:,i_cross],linewidth=2.5,label='Spectrum section')
            plt.plot(em_axis_mid,fit_cols[:,i_cross],label='Fitted baseline')
            # Mark the four fitting-region edges with vertical dotted lines
            c_idx = spike_row_mid
            nlo = c_idx - n_spike_buffer - n_fitting_buffer - 1
            nlo1 = c_idx - n_spike_buffer - 1
            nhi = c_idx + n_spike_buffer + 1
            nhi1 = c_idx + n_spike_buffer + n_fitting_buffer + 1

            plt.axvline(x=em_axis_mid[nlo], color='k', linestyle=':', linewidth=1.2, alpha=0.9, label='Fit data boundaries')
            for idx in (nlo1, nhi, nhi1):
                plt.axvline(x=em_axis_mid[idx], color='k', linestyle=':', linewidth=1.2, alpha=0.9)
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

def rl_deconvolve_nonneg(y, h, n_iter=200, pad_factor=2.0, eps=1e-12, smooth_sigma=0.0):
    """Richardson–Lucy deconvolution enforcing non-negativity.

    Model: y ≈ h * x (linear convolution), with x ≥ 0.

    Args:
        y: observed 1D signal (real, preferably non-negative)
        h: kernel (impulse response); must have sum(h) > 0
        n_iter: RL iterations (typical 50–300)
        pad_factor: multiplier for linear conv length before next pow2 (reduces wrap-around)
        eps: small constant to avoid divide-by-zero
        smooth_sigma: optional Gaussian smoothing each iteration (in samples) to stabilize
        center_output: roll so max is centered then crop back to len(y)

    Returns:
        x_est: non-negative deconvolved estimate of length len(y)
    """
    y = np.asarray(y, float)
    h = np.asarray(h, float)
    n = y.size
    m = h.size
    if n == 0 or m == 0:
        return np.array([])
    h_sum = np.sum(h)
    if h_sum <= 0:
        raise ValueError("Kernel must have positive sum for RL deconvolution")
    h = h / h_sum  # flux preservation
    lin_len = n + m - 1
    N_fft = int(2 ** np.ceil(np.log2(lin_len * pad_factor)))
    y_pad = np.zeros(N_fft)
    h_pad = np.zeros(N_fft)
    y_pad[:n] = y
    h_pad[:m] = h
    H = np.fft.fft(h_pad)
    H_flip = np.fft.fft(h_pad[::-1])
    x = np.maximum(y_pad, 0.0) + eps  # init
    for _ in range(int(n_iter)):
        conv_x = np.fft.ifft(H * np.fft.fft(x)).real
        ratio = y_pad / (conv_x + eps)
        corr = np.fft.ifft(H_flip * np.fft.fft(ratio)).real
        x *= corr
        x = np.maximum(x, 0.0)
        if smooth_sigma > 0.0:
            from scipy.ndimage import gaussian_filter1d
            x = gaussian_filter1d(x, smooth_sigma, mode='nearest')
            x = np.maximum(x, 0.0)
    x = np.fft.fftshift(x)
    return x[N_fft//2-n//2:N_fft//2+n//2]

def _roll_center_max(a):
    a = np.asarray(a)
    if a.size == 0:
        return a
    max_idx = int(np.argmax(a))
    center_idx = a.size // 2
    return np.roll(a, center_idx - max_idx)

def synth_baseline(sp_pr,sp_x,om_pr,om_x,T):

    phase0 = np.exp(1j*0*T)
    phase1 = np.exp(1j*om_pr*T)
    phase2 = np.exp(-1j*om_x*T)

    f1 = sp_pr * phase0
    f2 = -sp_x/om_x * phase2
    conv1 = fftconvolve(f1,f2,mode='same',axes=1)

    f1 = -sp_pr/om_pr * phase1
    f2 = sp_x * phase0
    conv2 = fftconvolve(f1,f2,mode='same',axes=1)

    return conv1 + conv2

def synth_baseline_hermit(input,sp_x,om_pr,om_x,T):

    phase0 = np.exp(1j*0*T)
    phase1 = np.exp(1j*om_pr*T)
    phase2 = np.exp(-1j*om_x*T)

    f1 = input
    f2 = -sp_x/om_x * np.conjugate(phase2)
    conv1 = fftconvolve(f1,f2[:,::-1],mode='same',axes=1)

    f1 = input
    f2 = sp_x * phase0
    conv2 = fftconvolve(f1,f2[:,::-1],mode='same',axes=1)
    conv2 = -conv2/om_pr * np.conjugate(phase1)

    return np.sum(conv1 + conv2, axis=0)

# def synth_baseline_hermit(d,bi,omd,ombi,T):

#     phase0 = np.exp(1j*0*T)
#     phase1 = np.exp(1j*om1*T)
#     phase2 = np.exp(-1j*om2*T)

#     conv1 = fftconvolve(d*phase0,(bi*phase0)[:,::-1],mode='same',axes=1)
#     conv1 = (-phase1/omd)*conv1

#     bi_mod = bi*(-phase2/ombi)
#     conv2 = fftconvolve(d*phase0,bi_mod[:,::-1],mode='same',axes=1)

#     return conv1 + conv2

def synth_baseline_tau(sp1,sp2,om1,om2,tau):

    phase1 = np.exp(1j*om1*tau)
    phase2 = np.exp(-1j*om2*tau)

    f1 = -(phase1/om1) * sp1
    f2 = sp2
    conv = fftconvolve(f1,f2,mode='same')

    # f1 = sp1
    # f2 = -sp2/om2 * phase2
    # conv += fftconvolve(f1,f2,mode='same')

    return conv

def synth_baseline_transpose_tau(sp1,sp2,om1,om2,tau):

    phase1 = np.exp(-1j*om1*tau)
    phase2 = np.exp(1j*om2*tau)

    conv = fftconvolve(sp1,sp2[::-1],mode='same')
    conv = (-phase1/om1) * conv

    # sp2_mod = sp2*(-phase2/om2)
    # conv += fftconvolve(sp1,sp2_mod[::-1],mode='same')

    return conv

def plotc(ar):
    plt.plot(np.abs(ar))
    plt.plot(np.real(ar),linewidth=0.8)
    plt.plot(np.imag(ar),linewidth=0.8)
    plt.show()

def reconstruct_WirtFlow(sig_measrd,sp_probe,sp_xuv,om_probe,om_xuv,T,
                         n_power_iter=50,n_main_iter=10000,mu_step_max=0.06,I_warmup=1000,ifplot=50):

    n_e, n_t = sig_measrd.shape

    w = np.ones((n_e,)).astype(complex) / np.sqrt(n_e)

    m = np.size(sig_measrd)

    for i_iter in range(n_power_iter):

        w = 1/m * synth_baseline_hermit(sig_measrd * synth_baseline(w,sp_xuv,om_probe,om_xuv,T),sp_xuv,om_probe,om_xuv,T)
        w = w / np.sqrt(np.sum(np.abs(w)**2))

    lambda_bsln = synth_baseline(w,sp_xuv,om_probe,om_xuv,T)
    alpha = np.sum( np.sqrt(sig_measrd) * np.abs(lambda_bsln) ) / np.sum( np.abs(lambda_bsln)**2 )

    w = alpha*w

    plt.plot(np.real(w),linewidth=0.7)
    plt.plot(np.imag(w),linewidth=0.7)
    plt.plot(np.abs(sp_probe))
    plt.savefig('initial_spectrum_estimate.png')
    plt.close()

    normsq_z0 = np.sum( np.abs(w)**2 )
    z = np.copy(w)

    ers = []

    for i_iter in range(n_main_iter):

        mu_step = mu_step_max*(1-np.exp(-(i_iter+1)/I_warmup))

        ers.append( np.sum(np.abs( np.abs(z) - np.abs(sp_probe) )**2)/np.sum(np.abs(sp_probe)**2) )

        sbforward = synth_baseline(z,sp_xuv,om_probe,om_xuv,T)
        sbhermit_arg = (np.abs(sbforward)**2 - sig_measrd) * sbforward
        z = z - mu_step/normsq_z0*1/m * synth_baseline_hermit(sbhermit_arg,sp_xuv,om_probe,om_xuv,T)

        print(i_iter)

        if i_iter%int(ifplot)==0 and bool(ifplot):
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))
            # axes[0].plot(np.real(z), label='Re(z)')
            # axes[0].plot(np.imag(z), label='Im(z)')
            # axes[0].plot(np.real(sp_probe), label='Re(sp_probe)')
            # axes[0].plot(np.imag(sp_probe), label='Im(sp_probe)')

            axes[0].plot(np.abs(z), label='Abs(z)')
            axes[0].plot(np.abs(sp_probe), label='Abs(sp_probe)')
            # axes[0].plot(np.angle(z), label='Abs(z)')
            # axes[0].plot(np.angle(sp_probe), label='Abs(sp_probe)')

            axes[0].set_title('Current spectrum estimate')
            axes[0].legend()

            axes[1].plot(ers, label='Relative MSE')
            axes[1].set_yscale('log')
            axes[1].set_title('Error history')
            axes[1].set_xlabel('Recorded iteration')
            axes[1].legend()
            if ers:
                last_val = ers[-1]
                axes[1].text(
                    0.02, 0.1,
                    f'Iter {i_iter}: {last_val:.5f}',
                    transform=axes[1].transAxes,
                    fontsize=10,
                    color='white',
                    weight='bold',
                    ha='left',
                    va='top',
                    path_effects=[pe.withStroke(linewidth=1, foreground='black')]
                )

            plt.tight_layout()
            plt.savefig('spectrum_convergence_iter.png')
            plt.close()

    return z

###
### FIELD PARAMETERS
###

E_lo = 60.0
E_hi = 63.5
T_reach = 100
E_span = E_hi-E_lo

N_E = 1000
N_T = 1000

E_range = np.linspace(E_lo,E_hi,N_E)
T_range = np.linspace(-T_reach,T_reach,N_T)
E, T = np.meshgrid(E_range,T_range)

A_xuv = 1
om_xuv = 60.65/hbar
s_xuv = 0.15/hbar
pulse_xuv = (0*T,A_xuv,om_xuv,s_xuv,0)

A_probe = 1.0
om_probe = 1.55/hbar
s_probe = 0.15/hbar
pulse_probe = (T,A_probe,om_probe,s_probe,0)

A_probe2 = 0.2
om_probe2 = 1.20/hbar
s_probe2 = 0.04/hbar
pulse_probe2 = (T,A_probe2,om_probe2,s_probe2,0)

A_probe3 = 0.2
om_probe3 = 2.0/hbar
s_probe3 = 0.07/hbar
pulse_probe3 = (T,A_probe3,om_probe3,s_probe3,0)

A_probe4 = 0.3
om_probe4 = 1.85/hbar
s_probe4 = 0.17/hbar
pulse_probe4 = (T,A_probe4,om_probe4,s_probe4,0)

A_ref = 1
om_ref = 1.55/hbar
s_ref = 0.005/hbar
pulse_ref = (0*T,A_ref,om_ref,s_ref,0)

refs = (pulse_ref,)
probes = (pulse_probe,pulse_probe2,pulse_probe3,pulse_probe4)
xuvs = (pulse_xuv,)
refprobes = tuple(list(refs) + list(probes))

###
### GENERATE SIGNAL
###

amplit_tot_0 = Amplitude(xuvs,refprobes,E)

signal_clean = np.abs(amplit_tot_0)**2

ifnoise = True

SNR = 20

if SNR is None or SNR <= 0 or not ifnoise:
    signal = signal_clean.copy()
else:
    sig_rms = np.sqrt(np.mean(signal_clean**2))
    noise_rms = sig_rms / float(SNR)
    noise = noise_rms * np.random.normal(size=signal_clean.shape)
    signal = signal_clean + noise

# if not ifnoise:
#     signal = signal_clean.copy()
# else:
#     # Apply Poisson noise assuming signal_clean contains expected values (means)
#     rng = np.random.default_rng()
#     signal = rng.poisson(signal_clean).astype(float)

amplit_tot_FT, OM_T, em_lo, em_hi = CFT(T_range,signal,use_window=False)

###
### CONTROL SAMPLE - DECONVOLUTION USING EXACT DATA
###

# Synthetic baseline with known spectra
om_probe = (E/hbar - E_lo/hbar + 0.1 + E_span/hbar/N_E/2 * ((N_E-1) % 2))[0,:]
sp_probe = sp_tot(probes,om_probe)
sp_ref = sp_tot(refs,om_probe)


om_xuv = (E/hbar - E_span/hbar/2 - 0.1)[0,:]
sp_xuv = sp_tot(xuvs,om_xuv)

# synth_bsln = np.abs(synth_baseline(sp_probe,sp_xuv,om_probe,om_xuv,T) + synth_baseline(sp_ref,sp_xuv,om_probe,om_xuv,T*0))**2
synth_bsln = np.abs(synth_baseline(sp_probe,sp_xuv,om_probe,om_xuv,T))**2
synth_bsln_FT, _, _, _ = CFT(T_range,synth_bsln,use_window=False)
# CHECKED - PRODUCES SAME AS 'AMPLITUDE'

# sp_rec = reconstruct_WirtFlow(synth_bsln,sp_probe,sp_xuv,om_probe,om_xuv,T)

###
### DETRENDING - SEPARATING PROBE AND REF ZERO FREQ COMPONENT 
###

em_axis_mid_reach = 1.00
slice_fracts = (0.5*(1 - em_axis_mid_reach/em_hi), 0.5*(1 + em_axis_mid_reach/em_hi))

amplit_tot_FT_mid, em_axis_mid, i0, i1 = extract_midslice(amplit_tot_FT, slice_fracts, hbar*OM_T[:,0])

mask_em = 0.86
mask_frac = 0.69

N_em = np.size(em_axis_mid)
i_mask_em = floor(0.5*(1-(mask_em/em_axis_mid_reach))*N_em)

amplit_tot_FT_mid[0:i_mask_em,0:floor(mask_frac*N_E)] = 0
amplit_tot_FT_mid[N_em-i_mask_em+1:N_em,0:floor(mask_frac*N_E)] = 0

amplit_tot_FT_mid_detrended, spike_only, spike_row_mid = detrend_spike(amplit_tot_FT_mid,em_axis_mid,0,2,plot=False)

# Zero-pad the detrended mid-band to the full (ω, E) grid
amplit_tot_FT_detrended_full = np.zeros_like(amplit_tot_FT, dtype=complex)
amplit_tot_FT_detrended_full[i0:i1, :] = amplit_tot_FT_mid_detrended

sig_probe_reconstructed,_,_,_ = CFT(T_range,amplit_tot_FT_detrended_full,use_window=False,inverse=True)


### Check quality of detrending

meas_baseline = np.sqrt(np.abs(sig_probe_reconstructed))
analytic_bsln = np.abs(Amplitude(xuvs,probes,E))**2
analytic_bsln_FT,_,_,_ = CFT(T_range,analytic_bsln,use_window=False)

# plot_mat(analytic_bsln_FT - amplit_tot_FT_detrended_full)

spike_only = spike_only[spike_row_mid,:]

spike_only = np.sqrt(np.abs(spike_only))

# Align spike_only's maximum (along energy) with sp_xuv's maximum
idx_xuv = int(np.argmax(sp_xuv))
idx_spk = int(np.argmax(spike_only))
shift_cols = idx_xuv - idx_spk
spike_only = np.roll(spike_only, shift_cols)

spike_only = spike_only*np.max(sp_xuv)/np.max(spike_only)

# --- Decompose RL reconstruction as a sum of n Gaussians and plot ---
n_comp = 4
# Use non-negative target for Gaussian fit

fit_gauss, fit_params = fit_n_gaussians_1d(
    y_vals=om_xuv,
    z_vals=spike_only,
    n=n_comp,
    return_params=True,
    return_components=False,
    use_weights=False,
)

plt.plot(sp_xuv)
plt.plot(spike_only,linewidth=0.7)
plt.plot(fit_gauss,linewidth=0.7)
plt.show()


###
### RETRIEVE PROBE SPECTRUM
###

### Prepare

meas_baseline = meas_baseline**2
meas_baseline = meas_baseline*np.max(synth_bsln)/np.max(meas_baseline)

plot_mat(meas_baseline)
plot_mat(synth_bsln)


plot_mat((meas_baseline - synth_bsln)/np.max(meas_baseline))

sp_rec = reconstruct_WirtFlow(meas_baseline,sp_probe,spike_only,om_probe,om_xuv,T,n_power_iter=150,n_main_iter=10000,mu_step_max=0.01,I_warmup=300,ifplot=50)