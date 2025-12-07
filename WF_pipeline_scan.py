import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from math import floor
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from matplotlib import patheffects as pe
from skimage.restoration import denoise_tv_bregman
from scipy.ndimage import median_filter
from scipy.special import i0e, i1e
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import sqrtm

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
        out += A/(2*s) * np.exp(-0.5 * ((y - mu) / s) ** 2)
    return out

def fit_n_gaussians_1d(
    y_vals,
    z_vals,
    n,
    use_weights=False,
    width_frac_bounds=(0.01, 0.6),   # min/max sigma as fraction of span_y
    amp_min=0.0                      # amplitude lower bound
):
    """
    Fit a sum of n 1D Gaussians with reasonable bounds:
      - amplitudes >= amp_min
      - centers within [y_min, y_max]
      - widths within [min_s, max_s] where min_s/max_s are fractions of span_y

    The target shape is assumed concentrated around the middle of the array.
    """
    y_vals = np.asarray(y_vals, float)
    z_vals = np.asarray(z_vals, float)

    N = y_vals.size
    if N < 3:
        raise ValueError("y_vals must have at least 3 points")

    y_min, y_max = float(y_vals[0]), float(y_vals[-1])
    span_y = float(y_max - y_min)
    if span_y <= 0:
        raise ValueError("y_vals must be strictly increasing")

    # Initial guesses: place all centers at the peak, split amplitude, reasonable width
    peak_idx = int(np.argmax(z_vals))
    mu_peak = float(y_vals[peak_idx])
    A_tot = float(np.max(np.abs(z_vals))) if np.any(z_vals) else 1.0
    A0s = np.ones(n) * (A_tot / max(n, 1))
    mu0s = np.ones(n) * mu_peak
    s0 = 0.1 * span_y

    theta0 = []
    for k in range(int(n)):
        theta0.extend([float(A0s[k]), float(mu0s[k]), float(s0)])
    theta0 = np.asarray(theta0, float)

    # Bounds
    min_s = max(width_frac_bounds[0] * span_y, 1e-12)
    max_s = max(width_frac_bounds[1] * span_y, min_s * 2)

    lower = []
    upper = []
    for _ in range(int(n)):
        lower.extend([float(amp_min), y_min, min_s])
        upper.extend([np.inf,         y_max, max_s])
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)

    # Optional weights: Poisson-like -> sigma ~ sqrt(z + eps)
    sigma = None
    if use_weights:
        eps = 1e-12
        sigma = np.sqrt(np.abs(z_vals) + eps)
        sigma[sigma < 1e-12] = 1e-12

    try:
        popt, _ = curve_fit(
            _sum_n_gauss1d,
            y_vals,
            z_vals,
            p0=theta0,
            bounds=(lower, upper),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=200000
        )
        fit_full = _sum_n_gauss1d(y_vals, *popt)
    except Exception:
        # Fallback: return zeros with initial params
        popt = theta0
        fit_full = np.zeros_like(y_vals, dtype=float)

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

        for v in np.linspace(0,0.99,20):
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

def plotc(ar):
    plt.plot(np.abs(ar))
    plt.plot(np.real(ar),linewidth=0.8)
    plt.plot(np.imag(ar),linewidth=0.8)
    plt.show()

def nfit_params_to_probes(params):
    probes = []
    for k in range(len(params)//3):
        probes.append((T,params[3*k],params[3*k+1],params[3*k+2],0))
    return tuple(probes)

def reconstruct_WirtFlow(sig_measrd,sp_probe,sp_xuv,om_probe,om_xuv,T,b_est,
                         n_power_iter=50,n_main_iter=10000,ifplot=50,naive_init=None,
                         median_regval=2,lastmax_margin=200,ifwait=True,alph=0,nt=0):

    n_t, n_e = sig_measrd.shape

    if naive_init is None:
        w = np.ones((n_e,)).astype(complex) / np.sqrt(n_e)
        poisson_wghts = np.copy(sig_measrd)
        poisson_wghts[sig_measrd < median_regval * np.median(sig_measrd)] = 0

        m = np.size(poisson_wghts)

        for i_iter in range(n_power_iter):
            w = 1/m * synth_baseline_hermit(poisson_wghts * synth_baseline(w,sp_xuv,om_probe,om_xuv,T),sp_xuv,om_probe,om_xuv,T)
            lambda_val0 = np.sqrt(np.sum(np.abs(w)**2))
            w = w / lambda_val0

        lambda_bsln = synth_baseline(w,sp_xuv,om_probe,om_xuv,T)
        alpha = np.sum( np.sqrt(poisson_wghts-b_est) * np.abs(lambda_bsln) ) / np.sum( np.abs(lambda_bsln)**2 )

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
    coarse_bin = 10

    for i_iter in range(n_main_iter):

        ers.append( np.sum(np.abs( normalize_abs(np.abs(z)) - normalize_abs(np.abs(sp_probe)) )**2)/np.sum(normalize_abs(np.abs(sp_probe))**2) )

        sbforward = synth_baseline(z,sp_xuv,om_probe,om_xuv,T)


        # sbhermit_arg = (np.abs(sbforward)**2 - sig_measrd) * sbforward  GAUSSIAN WF
        lambda_arr = np.abs(sbforward)**2 + b_est
        sbhermit_arg = (1 - sig_measrd/lambda_arr) * sbforward # POISSON WF (??)

        grad = synth_baseline_hermit(sbhermit_arg,sp_xuv,om_probe,om_xuv,T)

        weight_vec = sig_measrd/(lambda_arr)**2
        grad_forward = synth_baseline(grad,sp_xuv,om_probe,om_xuv,T)
        denom = np.sum(np.conjugate(grad_forward)*weight_vec*grad_forward)
        mu_step = np.sum(np.abs(grad)**2)/denom

        z = z - mu_step/normsq_z0 * grad

        mus.append(mu_step)

        # print(i_iter)

        if i_iter%int(ifplot)==ifplot-1 and bool(ifplot):
            # fig, axes = plt.subplots(2, 1, figsize=(8, 6))

            
            fig, axes = plt.subplots(3, 1, figsize=(8, 6))
            axes[2].plot(np.abs(mus))
            axes[2].set_yscale('log')

            # axes[0].plot(np.real(z), label='Re(z)')
            # axes[0].plot(np.imag(z), label='Im(z)')
            # axes[0].plot(np.real(sp_probe), label='Re(sp_probe)')
            # axes[0].plot(np.imag(sp_probe), label='Im(sp_probe)')

            axes[0].plot(normalize_abs(np.abs(z)), label='Abs(z)')
            axes[0].plot(normalize_abs(np.abs(sp_probe)), label='Abs(sp_probe)')
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
            plt.savefig('scans/ims/spectrum_convergence_iter%.2f_%i.png'%(alph,nt))
            plt.close()

        if i_iter%coarse_bin == coarse_bin-1:
            avgs = np.mean(np.abs(mus).reshape(-1, coarse_bin), axis=1)
            if i_iter - np.argmax(avgs)*10 > lastmax_margin and avgs[-1] < avgs[-2] and (not ifwait or avgs[-1] > avgs[0]) :
                break

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

    tau_x,A_x,om0_x,s_x,phi_x = pulse_params_x

    retval = 0
    A_tot = 0

    for probe in probes:
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
        retval += A_p/(s_pp) * np.exp(-0.5*((delta_p/s)**2 + (s_t * (om_t + delta_E))**2))

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

def project_to_density_matrix(M):
    """
    Project a measured matrix to the nearest valid density matrix 
    (Hermitian, PSD, trace 1) using eigenvalue thresholding.

    Parameters:
        M (np.ndarray): Measured matrix (complex)

    Returns:
        np.ndarray: Valid density matrix
    """
    # Step 1: Hermitize
    M = (M + M.conj().T) / 2

    # Step 2: Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(M)

    # Step 3: Threshold negative eigenvalues
    eigvals_clipped = np.clip(eigvals, 0, None)

    # If all eigenvalues are zero (extreme noise), avoid division by zero
    if np.sum(eigvals_clipped) == 0:
        # Return maximally mixed state
        d = M.shape[0]
        return np.eye(d) / d

    # Step 4: Renormalize eigenvalues to trace = 1
    eigvals_normalized = eigvals_clipped / np.sum(eigvals_clipped)

    # Step 5: Reconstruct density matrix
    rho = eigvecs @ np.diag(eigvals_normalized) @ eigvecs.conj().T

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

###
### FIELD PARAMETERS
###

alpha_range = np.array([0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64])
N_T_range = np.array([250,300,400,500,700,900,1100])

a,nt = np.meshgrid(alpha_range,N_T_range)

final_rec_sqerr = np.zeros_like(a)
fid1s = np.zeros_like(a)
fid2s = np.zeros_like(a)
fid3s = np.zeros_like(a)
fid4s = np.zeros_like(a)

for i_nt, N_T in enumerate(N_T_range):
    for i_a, alpha in enumerate(alpha_range):
        # try:
            E_lo = 60.0
            E_hi = 63.5
            T_reach = 100
            E_span = E_hi-E_lo

            E_res = 0.025
            N_E = round(E_span/E_res/10)*10

            ## In real polykraken data that I got, En_res is 0.025 eV, There are around 600 time delays and the count per pixel is up to 1e3 - 2e3 at max

            p_E = 4 # N_E upsampling integer

            b = 1

            E_range = np.linspace(E_lo,E_hi,N_E)
            T_range = np.linspace(-T_reach,T_reach,N_T)
            E, T = np.meshgrid(E_range,T_range)

            N_E_up = p_E*N_E
            E_up_range = np.linspace(E_lo,E_hi,N_E_up)
            E_up, T_up = np.meshgrid(E_up_range,T_range)

            A_xuv = 1
            om0_xuv = 60.65/hbar
            s_xuv = 0.15/hbar
            pulse_xuv = (0*T,A_xuv,om0_xuv,s_xuv,0)

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

            A_ref = 0.3
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

            ifnoise = True
            rng = np.random.default_rng()

            # Synthetic baseline with known spectra
            # om_probe = (E/hbar - E_lo/hbar + 0.1 + E_span/hbar/N_E/2 * ((N_E-1) % 2))[0,:]
            # om_xuv = (E/hbar - E_span/hbar/2 - 0.1)[0,:]
            om_probe = (E/hbar - E_lo/hbar + E_span/hbar/N_E/2 * ((N_E-1) % 2))[0,:]
            om_xuv = (E/hbar - E_span/hbar/2)[0,:]

            om_probe += 5*(om_probe[1] - om_probe[0])
            om_xuv += 5*(om_probe[0] - om_probe[1])

            sp_xuv = sp_tot(xuvs,om_xuv)
            sp_probe = sp_tot(probes,om_probe)
            sp_ref = sp_tot(refs,om_probe)

            om_probe_up = (E_up/hbar - E_lo/hbar + 0.1 + E_span/hbar/N_E_up/2 * ((N_E_up-1) % 2))[0,:]
            om_xuv_up = (E_up/hbar - E_span/hbar/2 - 0.1)[0,:]
            sp_xuv_up = sp_tot(xuvs,om_xuv_up)
            sp_probe_up = sp_tot(probes,om_probe_up)
            sp_ref_up = sp_tot(refs,om_probe_up)

            ### PROBE ONLY FOR REFERENCE
            synth_bsln = np.abs(downsample(synth_baseline(sp_probe_up,sp_xuv_up,om_probe_up,om_xuv_up,T_up),p_E))**2
            synth_bsln = rng.poisson(alpha*(synth_bsln)+b).astype(float) - b
            synth_bsln_FT, _, _, _ = CFT(T_range,synth_bsln,use_window=False)

            ### SYNTHETIC FULL SIGNAL
            amplit_tot_0 = downsample(synth_baseline(sp_probe_up,sp_xuv_up,om_probe_up,om_xuv_up,T_up) + synth_baseline(sp_ref_up,sp_xuv_up,om_probe_up,om_xuv_up,T_up*0),p_E)
            signal_clean = np.abs(amplit_tot_0)**2
            signal_clean *= alpha
            signal_clean += b

            if not ifnoise:
                signal = signal_clean.copy()
            else:
                # Apply Poisson noise assuming signal_clean contains expected values (means)
                signal = rng.poisson(signal_clean).astype(float)

            ### Total number of counts (signal/noise measure):

            # print("N Total = %.2e"%np.sum(signal))

            ###
            ### PROCESSING STARTS HERE
            ###

            noise_area_Elo, noise_area_Ehi = 0.0,0.3

            b_est = np.mean(signal[:,floor(noise_area_Elo*N_E):floor(noise_area_Ehi*N_E)])

            signal -= b_est

            # plot_mat(signal)

            amplit_tot_FT, OM_T, em_lo, em_hi = CFT(T_range,signal,use_window=False)
            amplit_tot_FT_wndw, OM_T, em_lo, em_hi = CFT(T_range,signal,use_window=True)

            # import sgolay2
            # sg2 = sgolay2.SGolayFilter2(window_size=13,poly_order=3)
            # signal = sg2(signal)

            ###
            ### DETRENDING - SEPARATING PROBE AND REF ZERO FREQ COMPONENT 
            ###

            em_axis_mid_reach = 1.00
            slice_fracts = (0.5*(1 - em_axis_mid_reach/em_hi), 0.5*(1 + em_axis_mid_reach/em_hi))

            amplit_tot_FT_mid, em_axis_mid, i0, i1 = extract_midslice(amplit_tot_FT, slice_fracts, hbar*OM_T[:,0])

            amplit_tot_FT_mid_detrended, spike_only, spike_row_mid = detrend_spike(amplit_tot_FT_mid,em_axis_mid,0,2,plot=False)

            # Zero-pad the detrended mid-band to the full (ω, E) grid
            amplit_tot_FT_detrended_full = np.copy(amplit_tot_FT)
            amplit_tot_FT_detrended_full[i0:i1, :] = amplit_tot_FT_mid_detrended

            # Extent the poissonian noise with rician bias

            sidebands_reach = 2.15
            sideband_lo, sideband_hi = floor(0.5*(1 - sidebands_reach/em_hi)*N_T), floor(0.5*(1 + sidebands_reach/em_hi)*N_T)
            amplit_tot_FT_detrended_full[sideband_lo:i0,:] = amplit_tot_FT[0:i0-sideband_lo,:]
            amplit_tot_FT_detrended_full[i1:sideband_hi,:] = amplit_tot_FT[N_T-sideband_hi+i1:,:]

            ###
            ### KOAY-BASSER FULL SIGNAL CORRECTION
            ###

            tot_rician_full = np.sum(signal+b_est,axis=0)
            z_target = np.sum(signal,axis=0)

            # # --- Decompose RL reconstruction as a sum of n Gaussians and plot ---
            # n_comp = 12
            # # Use non-negative target for Gaussian fit
            # tot_rician_pr_fit, _ = fit_n_gaussians_1d(
            #     y_vals=om_probe,
            #     z_vals=z_target,
            #     n=n_comp
            # )
            # tot_rician_full_fit = tot_rician_pr_fit + b_est*N_T

            amp_corr = np.zeros_like(amplit_tot_FT_wndw)

            for j in range(N_E):
                col_now = np.abs(amplit_tot_FT_wndw[:,j])
                # bias_now = tot_rician_full_fit[j]
                bias_now = tot_rician_full[j]
                amp_corr_now = koay_basser_correction(col_now,bias_now,lambda_thresh=1)
                amp_corr[:,j] = amp_corr_now

            phase = np.angle(amplit_tot_FT_wndw)
            amplit_tot_FT_wndw = amp_corr * np.exp(1j*phase)

            # plot_mat(amplit_tot_FT_wndw+1e-20)

            ###
            ### KOAY-BASSER PROBE SIGNAL CORRECTION
            ###

            # mixed signal time reach
            T_mix_reach = 40
            i_mix = floor(T_mix_reach/T_reach/2*N_T)

            # Remove rows [N_T//2 - i_mix : N_T//2 + i_mix] along time axis
            signal_xuv = np.delete(signal, np.s_[N_T//2 - i_mix : N_T//2 + i_mix], axis=0)

            tot_rician = np.mean(signal_xuv,axis=0)*N_T*(2*T_reach/N_T)**2
            # tot_rician = (np.mean(signal[:avg_lim1,:],axis=0) + np.mean(signal[avg_lim2:,:],axis=0))/2*N_T*(2*T_reach/N_T)**2

            # --- Decompose RL reconstruction as a sum of n Gaussians and plot ---
            n_comp = 1
            # Use non-negative target for Gaussian fit
            tot_rician_fit, _ = fit_n_gaussians_1d(
                y_vals=om_xuv,
                z_vals=tot_rician,
                n=n_comp
            )

            amp_corr = np.zeros_like(amplit_tot_FT_detrended_full)

            for j in range(N_E):
                col_now = np.abs(amplit_tot_FT_detrended_full[:,j])
                bias_now = tot_rician_fit[j]
                amp_corr_now = koay_basser_correction(col_now,bias_now)
                amp_corr[:,j] = amp_corr_now

            phase = np.angle(amplit_tot_FT_detrended_full)
            amplit_tot_FT_detrended_full = amp_corr * np.exp(1j*phase)

            # pow_ana = np.abs(synth_bsln_FT)**2
            # pow_mes = np.abs(amplit_tot_FT_detrended_full)**2
            # plt.plot(np.mean(pow_ana[:1000,:],axis=0))
            # plt.plot(np.mean(pow_mes[:1000,:],axis=0))
            # plt.show()

            sig_probe_reconstructed,_,_,_ = CFT(T_range,amplit_tot_FT_detrended_full,use_window=False,inverse=True)

            # plot_mat(sig_probe_reconstructed)
            # plot_mat(synth_bsln)
            # plot_mat((sig_probe_reconstructed - synth_bsln)/np.max(sig_probe_reconstructed))

            ###
            ### XUV PEAK
            ###

            sp_xuv_meas_sig_fit = np.sqrt(tot_rician_fit)
            sp_xuv_meas_sig_fit *= np.max(sp_xuv)/np.max(sp_xuv_meas_sig_fit)

            # Align fit_gauss's maximum (along energy) with sp_xuv's maximum
            idx_xuv = int(np.argmax(sp_xuv))
            idx_spk = int(np.argmax(sp_xuv_meas_sig_fit))
            shift_cols = idx_xuv - idx_spk
            sp_xuv_meas_sig_fit = np.roll(sp_xuv_meas_sig_fit, shift_cols)


            # --- Decompose RL reconstruction as a sum of n Gaussians and plot ---
            n_comp = 1
            # Use non-negative target for Gaussian fit
            sp_xuv_meas_sig_fit, sp_xuv_meas_sig_fit_params = fit_n_gaussians_1d(
                y_vals=om_xuv,
                z_vals=sp_xuv_meas_sig_fit,
                n=n_comp
            )

            xuvs_rec = nfit_params_to_probes(sp_xuv_meas_sig_fit_params)

            # plt.plot(sp_xuv,label='true spec')
            # plt.plot(sp_xuv_meas_sig_fit,linewidth=0.65,linestyle='--',label='sig rec')
            # plt.legend()
            # plt.show()


            ###
            ### RETRIEVE PROBE SPECTRUM
            ###

            file = './rec_spectra/sp_probe.npy'

            sig_probe_reconstructed = sig_probe_reconstructed + b_est

            sp_rec, rec_sqerr = reconstruct_WirtFlow(sig_probe_reconstructed,sp_probe,sp_xuv_meas_sig_fit,om_probe,om_xuv,T,b_est,
                                                     n_power_iter=50,n_main_iter=3000,ifplot=50,median_regval=4,lastmax_margin=np.sqrt(alpha)*700,
                                                     ifwait=False,alph=alpha,nt=N_T)
            np.save(file,sp_rec)

            sp_rec = np.load(file)

            final_rec_sqerr[i_nt,i_a] = rec_sqerr
            np.save('scans/sqerrs.npy',final_rec_sqerr)

            # --- Decompose RL reconstruction as a sum of n Gaussians and plot ---
            n_comp = 10
            om_grid = om_probe
            # Use non-negative target for Gaussian fit

            z_target = np.abs(sp_rec)

            fit_gauss, fit_params = fit_n_gaussians_1d(
                y_vals=om_grid,
                z_vals=z_target,
                n=n_comp
            )

            probes_reconstructed = nfit_params_to_probes(fit_params)

            plt.figure()
            plt.plot(om_grid*hbar, normalize_abs(z_target), label='WF target')
            plt.plot(om_grid*hbar, normalize_abs(fit_gauss), label=f'{n_comp}-Gaussian fit')
            plt.plot(om_grid*hbar, normalize_abs(sp_probe), label='True spectrum')
            plt.xlabel('E - E0')
            plt.ylabel('Amplitude (normalized)')
            plt.title('n-Gaussian decomposition of RL reconstruction')
            plt.legend()
            plt.tight_layout()
            plt.savefig('final_rec_probe.png',dpi=300)
            plt.close()

            dzeta_val = 1e-3
            theta_val = 0.01

            correction = correcting_function_multi(OM_T,E,normalize_params(xuvs,om_xuv)[0],normalize_params(probes,om_probe),dzeta=dzeta_val,theta=theta_val)
            amplit_tot_FT_corrected = correction*amplit_tot_FT_wndw
            amplit_tot_FT_corrected = median_filter(np.abs(amplit_tot_FT_corrected),size=(3,3))*np.exp(1j*np.angle(amplit_tot_FT_corrected))

            correction_x = correcting_function_multi(OM_T,E,normalize_params(xuvs_rec,om_xuv)[0],normalize_params(probes,om_probe),dzeta=dzeta_val,theta=theta_val)
            amplit_tot_FT_corrected_x = correction_x*amplit_tot_FT_wndw
            amplit_tot_FT_corrected_x = median_filter(np.abs(amplit_tot_FT_corrected_x),size=(3,3))*np.exp(1j*np.angle(amplit_tot_FT_corrected_x))

            correction_rec_x = correcting_function_multi(OM_T,E,normalize_params(xuvs_rec,om_xuv)[0],normalize_params(probes_reconstructed,om_probe),dzeta=dzeta_val,theta=theta_val)
            amplit_tot_FT_corrected_rec_x_nomedian = correction_rec_x*amplit_tot_FT_wndw
            amplit_tot_FT_corrected_rec_x = median_filter(np.abs(amplit_tot_FT_corrected_rec_x_nomedian),size=(3,3))*np.exp(1j*np.angle(amplit_tot_FT_corrected_rec_x_nomedian))


            ###
            ### RESAMPLE AND ANALYZE
            ###

            rho_lo = 59.6
            rho_hi = 61.4

            rho_reconstructed, amplit_tot_FT_corrected_small, extent_small, idxs_small, E1, E2 = resample(amplit_tot_FT_corrected,rho_hi,rho_lo,om_ref,E,OM_T,N_T)
            rho_reconstructed = project_to_density_matrix(rho_reconstructed)

            rho_reconstructed_x, amplit_tot_FT_corrected_small, extent_small, idxs_small, _, _ = resample(amplit_tot_FT_corrected_x,rho_hi,rho_lo,om_ref,E,OM_T,N_T)
            rho_reconstructed_x = project_to_density_matrix(rho_reconstructed_x)

            rho_reconstructed_rec_x, amplit_tot_FT_corrected_small_rec, extent_small, idxs_small, _, _ = resample(amplit_tot_FT_corrected_rec_x,rho_hi,rho_lo,om_ref,E,OM_T,N_T)
            rho_reconstructed_rec_x = project_to_density_matrix(rho_reconstructed_rec_x)

            rho_reconstructed_rec_x_nomedian, amplit_tot_FT_corrected_small_rec, extent_small, idxs_small, _, _ = resample(amplit_tot_FT_corrected_rec_x_nomedian,rho_hi,rho_lo,om_ref,E,OM_T,N_T)
            rho_reconstructed_rec_x_nomedian = project_to_density_matrix(rho_reconstructed_rec_x_nomedian)

            plot_mat(rho_reconstructed,extent=[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',
                    mode='abs',saveloc='rhos/synth_corr.png',xlabel='Energy [eV]',ylabel='Energy [eV]',
                    title='Rho ideally corrected for the probe spectrum',show=False)

            plot_mat(rho_reconstructed_x,extent=[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',
                    mode='abs',saveloc='rhos/synth_corr_x.png',xlabel='Energy [eV]',ylabel='Energy [eV]',
                    title='Rho rec xuv and ideally corrected for the probe spectrum',show=False)

            plot_mat(rho_reconstructed_rec_x,extent=[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',
                    mode='abs',saveloc='rhos/rec_corr_x.png',xlabel='Energy [eV]',ylabel='Energy [eV]',
                    title='Rho rec xuv and WF rec probe',show=False)

            plot_mat(rho_reconstructed_rec_x_nomedian,extent=[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',
                    mode='abs',saveloc='rhos/rec_corr_x_nomedian.png',xlabel='Energy [eV]',ylabel='Energy [eV]',
                    title='Rho rec xuv and WF rec probe',show=False)

            ###
            ### COMPARISON WITH A GROUND TRUTH
            ###

            ideal_rho = np.exp(-((E1 - om0_xuv*hbar)/s_xuv)**2 - ((E2 - om0_xuv*hbar)/s_xuv)**2)
            ideal_rho = project_to_density_matrix(ideal_rho)

            fid1 = fidelity(ideal_rho, rho_reconstructed)
            fid2 = fidelity(ideal_rho, rho_reconstructed_x)
            fid3 = fidelity(ideal_rho, rho_reconstructed_rec_x)
            fid4 = fidelity(ideal_rho, rho_reconstructed_rec_x_nomedian)

            # print(fid1,fid2,fid3,fid4)

            fid1s[i_nt,i_a] = fid1
            fid2s[i_nt,i_a] = fid2
            fid3s[i_nt,i_a] = fid3
            fid4s[i_nt,i_a] = fid4

            np.save('scans/fid1s.npy',fid1s)
            np.save('scans/fid2s.npy',fid2s)
            np.save('scans/fid3s.npy',fid3s)
            np.save('scans/fid4s.npy',fid4s)
            print(alpha, N_T)
        # except KeyboardInterrupt as e:
        #     print(f"Error at alpha={alpha}, N_T={N_T}: {type(e).__name__}")
        #     exit()
        # except Exception as e:
        #     print(f"Error at alpha={alpha}, N_T={N_T}: {type(e).__name__}: {e}")
        #     pass