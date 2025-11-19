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

def correcting_function_multi(om_t,ene_Eg,pulse_params_x,probes,dzeta=1e-3):
    tau_x,A_x,om0_x,s_x,phi_x = pulse_params_x
    mod_plus = modulating_function_multi(om_t,ene_Eg,pulse_params_x,probes)
    err_area_plus = np.where(mod_plus<dzeta)
    mod_plus[err_area_plus] = 1
    mod2_plus = modulating_function_multi(om_t,ene_Eg,pulse_params_x,probes,big_sigma=1000)

    correction_plus = om_t*mod2_plus/mod_plus
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

    Sig_cc_cubic_mesh = 1/2*(Sig_cc_cubic_mesh + np.conjugate(Sig_cc_cubic_mesh.T))

    Sig_cc_cubic_mesh = Sig_cc_cubic_mesh/np.sum(np.diag(np.abs(Sig_cc_cubic_mesh)))

    return Sig_cc_cubic_mesh, small_sig, extent, [idy_min,idy_max,idx_min,idx_max]

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

def Amplitude_ij_num(spectrum_fun_i,support_i,spectrum_fun_j,support_j,tau_range_i,tau_range_j,E_range):
    n_t = np.size(tau_range_j)
    n_e = np.size(E_range)
    integral_ress = np.zeros((n_t,n_e)).astype(complex)

    lo_i, hi_i = support_i
    lo_j, hi_j = support_j

    limit_lo = lo_i + lo_j # INCORRECT, THIS IS THE LIMIT FOR ENE!! (works for usual parameters by accident) TODO
    limit_hi = hi_i + hi_j

    numerator_fun = lambda om, Ene, tau_i, tau_j: spectrum_fun_i(Ene/hbar - om) * np.exp(1j*(Ene/hbar - om)*tau_i) * spectrum_fun_j(om) * np.exp(1j*om*tau_j)

    for i_t in range(n_t):
        for i_e in range(n_e):

            E_val = E_range[i_e]
            ti_val = tau_range_i[i_t]
            tj_val = tau_range_j[i_t]

            regularization_val = numerator_fun(E_val/hbar, E_val, ti_val, tj_val)

            integrand_fun = lambda om: (numerator_fun(om,E_val,ti_val,tj_val) - regularization_val) / (om - E_val/hbar)

            integral_res, err = integrate.quad(integrand_fun,limit_lo,limit_hi,complex_func=True)
            integral_ress[i_t,i_e] = -1j*(integral_res + 
                                        hbar*regularization_val*np.log(np.abs((limit_hi-E_val/hbar)/(E_val/hbar-limit_lo))) - 
                                        1j*pi*regularization_val)
    return integral_ress

def plot_spectra(gausses, n_points=1000, sigma_span=6.0, title='Spectrum', saveloc=None):
    om_los, om_his = [], []
    for _, A, om0, s, _ in gausses:
        if s is None or s <= 0:
            continue
        om_los.append(om0 - sigma_span * s)
        om_his.append(om0 + sigma_span * s)

    om_min = min(om_los)
    om_max = max(om_his)

    om = np.linspace(om_min, om_max, n_points)
    y = sp_tot(gausses, om)

    plt.plot(om*hbar, y)
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'$A(\omega)$')
    plt.title(title)
    plt.tight_layout()
    if saveloc is not None:
        plt.savefig(saveloc)
    plt.show()

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
            plt.savefig('newims/section.png')
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

def nfit_params_to_probes(params):
    probes = []
    for k in range(len(params)//3):
        probes.append((T,params[3*k],params[3*k+1],params[3*k+2],0))
    return tuple(probes)

def _denoise_1d(a, win_frac=0.07, poly=3, thresh_sigma=3.5):
    a = np.asarray(a, float)
    n = a.size
    win = max(5, int(win_frac * n))
    if win % 2 == 0:
        win += 1
    win = min(win, n - (1 - n % 2))  # ensure valid odd length
    smooth = savgol_filter(a, win, poly, mode='interp')
    resid = a - smooth
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    sigma = 1.4826 * mad
    soft = np.sign(resid) * np.maximum(np.abs(resid) - thresh_sigma * sigma, 0.0)
    return smooth + soft

def _roll_center_max(a):
    a = np.asarray(a)
    if a.size == 0:
        return a
    max_idx = int(np.argmax(a))
    center_idx = a.size // 2
    return np.roll(a, center_idx - max_idx)

def regularization_pos(sig,range,denoise=True,gaussianize=True,rollmax=False,n_gauss=12,k_enhance=3,win_frac=0.07,poly=3,thresh_sigma=3.5):
    retsig = sig
    if denoise:
        retsig = _denoise_1d(retsig,win_frac=win_frac,poly=poly,thresh_sigma=thresh_sigma)
    if gaussianize:
        range_out = np.linspace(range[0],range[-1],k_enhance*np.size(range))
        retsig, _ = fit_n_gaussians_1d(
            y_vals=range,
            z_vals=sig,
            n=n_gauss,
            return_params=True,
            return_components=False,
            use_weights=False,
            resample_y=range_out
        )
    else:
        range_out = range

    if rollmax:
        retsig = _roll_center_max(retsig)

    return retsig, range_out

###
### FIELD PARAMETERS
###

E_lo = 60.0
E_hi = 63.5
T_reach = 100
E_span = E_hi-E_lo

N_E = 1500
N_T = 1500

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
xuvs = (pulse_xuv,)#,pulse_xuv2,pulse_xuv3]
refprobes = tuple(list(refs) + list(probes))

# plot_spectra(probes,title='Probe spectrum',saveloc='newims/spectrum.png')

###
### GENERATE SIGNAL
###

amplit_tot_0 = Amplitude(xuvs,refprobes,E)

# Transition dipole element (square modulus summed over ionization OAM channels) is approximated as T_i(E) = a_i*(E-E_0)

a_dipole_0 = 0.00

signal_clean = (1 + a_dipole_0*(E-om_xuv*hbar)) * np.abs(amplit_tot_0)**2

SNR = None

if SNR is None or SNR <= 0:
    signal = signal_clean.copy()
else:
    sig_rms = np.sqrt(np.mean(signal_clean**2))
    noise_rms = sig_rms / float(SNR)
    noise = noise_rms * np.random.normal(size=signal_clean.shape)
    signal = signal_clean + noise

amplit_tot_FT, OM_T, em_lo, em_hi = CFT(T_range,signal,use_window=False)

# plot_mat(signal,[E_lo,E_hi,-T_reach,T_reach],cmap='plasma',mode='phase')

# plt.plot(signal[:,np.argmax(np.sum(np.abs(signal),axis=0))])
# plt.show()

# plot_mat(np.clip(amplit_tot_FT,-1,1),[E_lo,E_hi,em_lo,em_hi],cmap='plasma',mode='phase')

# plt.plot(np.real(amplit_tot_FT[:,np.argmax(np.sum(np.abs(signal),axis=0))]))
# plt.plot(np.imag(amplit_tot_FT[:,np.argmax(np.sum(np.abs(signal),axis=0))]))
# plt.show()

###
### DETRENDING - SEPARATING PROBE AND REF ZERO FREQ COMPONENT 
###


# --- Fit and subtract slowly varying background in each column ---
# Build the y-axis (energy units ħ·ω) for the selected mid band

em_axis_mid_reach = 1.00
slice_fracts = (0.5*(1 - em_axis_mid_reach/em_hi), 0.5*(1 + em_axis_mid_reach/em_hi))

amplit_tot_FT_mid, em_axis_mid, i0, i1 = extract_midslice(amplit_tot_FT, slice_fracts, hbar*OM_T[:,0])

mask_em = 0.86
mask_frac = 0.69

N_em = np.size(em_axis_mid)
i_mask_em = floor(0.5*(1-(mask_em/em_axis_mid_reach))*N_em)

amplit_tot_FT_mid[0:i_mask_em,0:floor(mask_frac*N_E)] = 0
amplit_tot_FT_mid[N_em-i_mask_em+1:N_em,0:floor(mask_frac*N_E)] = 0

# plot_mat(np.clip(np.abs(amplit_tot_FT),0,10),[E_lo,E_hi,em_lo,em_hi],cmap='plasma',mode='abs',
#          saveloc='newims/full_comp.png',xlabel='Direct energy [eV]',ylabel='Indirect energy [eV]',title='Full signal spectrum')

# plot_mat(np.clip(np.abs(amplit_tot_FT_mid),0,10),[E_lo,E_hi,em_axis_mid[0],em_axis_mid[-1]],cmap='plasma',mode='abs',
#          saveloc='newims/zero_comp.png',xlabel='Direct energy [eV]',ylabel='Indirect energy [eV]',title='Zero-frequency area of spectrum')

amplit_tot_FT_mid_detrended, spike_only, spike_row_mid = detrend_spike(amplit_tot_FT_mid,em_axis_mid,0,2,plot=False)

extent_mid = [E_lo, E_hi, float(em_axis_mid[0]), float(em_axis_mid[-1])]

# plot_mat(amplit_tot_FT_mid_detrended,extent_mid,mode='abs',saveloc='newims/probepeak_only.png',
#          xlabel='Direct energy [eV]',ylabel='Indirect energy [eV]',title='XUV-PROBE peak only')
# plot_mat(spike_only,extent_mid,mode='abs',saveloc='newims/refpeak_only.png',
#          xlabel='Direct energy [eV]',ylabel='Indirect energy [eV]',title='XUV-REF peak only')

# Zero-pad the detrended mid-band to the full (ω, E) grid
amplit_tot_FT_detrended_full = np.zeros_like(amplit_tot_FT, dtype=complex)
amplit_tot_FT_detrended_full[i0:i1, :] = amplit_tot_FT_mid_detrended

sig_probe_reconstructed,_,_,_ = CFT(T_range,amplit_tot_FT_detrended_full,use_window=False,inverse=True)

# plot_mat(sig_probe_reconstructed,[E_lo,E_hi,-T_reach,T_reach],mode='abs',saveloc='newims/probe_tausig.png',
#          xlabel='Direct energy [eV]',ylabel='Time delay [fs]',title='FT of XUV-PROBE peak (delay-convolution signal)')

meas_baseline = np.sqrt(np.abs(sig_probe_reconstructed))

analytic_bsln = np.abs(Amplitude(xuvs,probes,E))**2
analytic_bsln_FT,_,_,_ = CFT(T_range,analytic_bsln,use_window=False)

plot_mat(analytic_bsln_FT - amplit_tot_FT_detrended_full)

# plt.plot(analytic_bsln_FT[:,floor(0.53*N_E)])
# plt.plot(amplit_tot_FT_detrended_full[:,floor(0.53*N_E)])
# plt.show()

###
### EXTRACT PROBE SPECTRUM
###

# sigg = np.abs(Amplitude(xuvs,probes,E))**2

# Find the row containing the global maximum magnitude and extract it

# spike_probe = np.real(sig_probe_reconstructed[N_T//2, :])
# spike_xuv = np.real(spike_only[spike_row_mid, :])

# spike_probe = np.sqrt(np.abs(spike_probe))*np.sign(spike_probe)
# spike_xuv = np.sqrt(np.abs(spike_xuv))*np.sign(spike_xuv)

# k_enhance = 1
# spike_probe, _ = regularization_pos(spike_probe,E_range,denoise=True,gaussianize=True,k_enhance=k_enhance)
# spike_xuv, obs_Erange = regularization_pos(spike_xuv,E_range,denoise=True,gaussianize=True,rollmax=True,k_enhance=k_enhance)

# # Construct more appropriate kernel: use spike_xuv  normalized
# kernel = normalize_abs(spike_xuv)
# obs = normalize_abs(spike_probe)

# spcent = _roll_center_max(sp_tot(xuvs,obs_Erange/hbar))

# plt.plot(obs_Erange,normalize_abs(kernel),label='Retrieved xuv spectrum')
# plt.plot(obs_Erange,normalize_abs(spcent),label='True xuv spectrum')
# plt.title('XUV+REF zero-frequency peak')
# plt.xlabel('Energy [eV]')
# plt.ylabel('Amplitude [arb. u.]')
# plt.legend()
# plt.savefig('newims/ref_peak.png')
# plt.show()

# plt.plot(E_range,normalize_abs(np.sqrt(np.abs(sigg[N_T//2, :]))),label=r'$A_{xuv} * A_{probe}$ exact')
# plt.plot(obs_Erange,k_enhance*obs,label=r'$A_{xuv} * A_{probe}$ retrieved')
# plt.title('XUV+PROBE zero-frequency peak')
# plt.xlabel('Energy [eV]')
# plt.ylabel('Amplitude [arb. u.]')
# plt.legend()
# plt.savefig('newims/probe_peak.png')
# plt.show()

# # Deconvolve (Wiener + RL for comparison). Use RL result downstream.
# spike_deconv_rl = rl_deconvolve_nonneg(obs, kernel, n_iter=20000, pad_factor=5.0, smooth_sigma=1.2)

# sp_reconstructed_rl = spike_deconv_rl*(obs_Erange/hbar - om_xuv)

# spprobe = sp_tot(probes,obs_Erange/hbar - om_xuv)

# plt.plot(normalize_abs(sp_reconstructed_rl), label='RL deconvolution reconstructed')
# plt.plot(normalize_abs(spprobe), label='True spectrum')
# plt.title('Deconvolved PROBE spectrum')
# plt.xlabel('Energy [eV]')
# plt.ylabel('Amplitude [arb. u.]')
# plt.legend()
# plt.savefig('newims/probe_reconstruction.png')
# plt.show()

###
### 2D RETRIEVAL
###

def synth_baseline(sp1,sp2,om1,om2,T):

    phase0 = np.exp(1j*0*T)
    phase1 = np.exp(1j*om1*T)
    phase2 = np.exp(-1j*om2*T)

    f1 = -sp1/om1 * phase1
    f2 = sp2 * phase0
    conv = fftconvolve(f1,f2,mode='same',axes=1)

    f1 = sp1 * phase0
    f2 = -sp2/om2 * phase2
    conv += fftconvolve(f1,f2,mode='same',axes=1)

    # x_full = np.linspace(OM_probe[0,0]+OM_xuv[0,0],OM_probe[0,-1]+OM_xuv[0,-1],2*N_E-1)
    # start = (N_E - 1) // 2
    # stop = start + N_E
    # x_conv = x_full[start:stop]

    return conv

def synth_baseline_transpose(d,bi,omd,ombi,T):

    phase0 = np.exp(1j*0*T)
    phase1 = np.exp(1j*om1*T)
    phase2 = np.exp(-1j*om2*T)

    conv1 = fftconvolve(d*phase0,(bi*phase0)[:,::-1],mode='same',axes=1)
    conv1 = (-phase1/omd)*conv1

    bi_mod = bi*(-phase2/ombi)
    conv2 = fftconvolve(d*phase0,bi_mod[:,::-1],mode='same',axes=1)

    return conv1 + conv2

def synth_baseline_tau(sp1,sp2,om1,om2,tau):

    phase1 = np.exp(1j*om1*tau)
    phase2 = np.exp(-1j*om2*tau)

    f1 = -sp1/om1 * phase1
    f2 = sp2
    conv = fftconvolve(f1,f2,mode='same')

    f1 = sp1
    f2 = -sp2/om2 * phase2
    conv += fftconvolve(f1,f2,mode='same')

    return conv

def synth_baseline_transpose_tau(d,bi,omd,ombi,tau):

    phased = np.exp(1j*omd*tau)
    phasebi = np.exp(-1j*ombi*tau)

    conv1 = fftconvolve(d,bi[::-1],mode='same')
    conv1 = (-phased/omd)*conv1

    bi_mod = bi*(-phasebi/ombi)
    conv2 = fftconvolve(d,bi_mod[::-1],mode='same')

    return conv1 + conv2

meas_baseline = np.sqrt(np.abs(sig_probe_reconstructed))

om1 = (E/hbar - E_lo/hbar + 0.1 + E_span/hbar/N_E/2 * ((N_E-1) % 2))[0,:]
sp1 = sp_tot(probes,om1)

om2 = (E/hbar - E_span/hbar/2 - 0.1)[0,:]
sp2 = sp_tot(xuvs,om2)

linear_tranform = lambda arr1: synth_baseline(arr1,sp2,om1,om2,T)

synth_bsln = np.abs(synth_baseline(sp1,sp2,om1,om2,T))


# plot_mat(np.clip(normalize_abs(synth_bsln),0,1e-5))
# plot_mat(np.clip(normalize_abs(meas_baseline),0,1e-5))

# plot_mat( (normalize_abs(synth_bsln) - normalize_abs(meas_baseline))/np.max(normalize_abs(meas_baseline)) )

synth_bsln = synth_bsln/np.max(np.abs(synth_bsln))*np.max(np.abs(meas_baseline))

plot_mat(synth_bsln)
plot_mat(meas_baseline)

plot_mat(synth_bsln - meas_baseline)

meas_baseline = meas_baseline.astype(complex)

retrieved_data = np.ones_like(meas_baseline[N_T//2,:]).astype(complex)*np.average(meas_baseline[N_T//2,:])

n_iterations = 300
eps = 1e-10
ones = np.ones_like(retrieved_data).astype(complex)

s = 35
weights = np.exp(-(T_range)**2/(s**2))
weights = weights/np.sum(weights)
plt.plot(weights)
plt.show()

for i in range(n_iterations):
    Numr = np.zeros_like(retrieved_data).astype(complex)
    Denom = np.zeros_like(retrieved_data).astype(complex)
    for i_t, tau in enumerate(T_range):
        Pred_image = synth_baseline_tau(retrieved_data,sp2,om1,om2,tau)
        # Pred_image = np.maximum(np.abs(Pred_image),eps)
        Pred_image = np.maximum(np.abs(Pred_image),eps)*np.exp(1j*np.angle(Pred_image))
        Ratio = meas_baseline[i_t,:]/Pred_image
        Numr_i = np.conj(synth_baseline_transpose_tau(Ratio,sp2,om1,om2,tau))
        Numr += Numr_i
        Denom += np.conj(synth_baseline_transpose_tau(ones,sp2,om1,om2,tau))
    # Denom = np.maximum(np.abs(Denom),eps)
    Denom = np.maximum(np.abs(Denom),eps)*np.exp(1j*np.angle(Denom))
    factor = Numr/Denom

    retrieved_data = retrieved_data * factor

    if i%20==0:
        plt.plot(om1,normalize_abs(np.real(retrieved_data)))
        plt.plot(om1,normalize_abs(sp1))
        plt.show()

# ###
# ### CORRECTION ANALYTICALLY
# ###

# # --- Decompose RL reconstruction as a sum of n Gaussians and plot ---
# n_comp = 12
# om_grid = obs_Erange/hbar - om_xuv
# # Use non-negative target for Gaussian fit
# z_target = np.maximum(sp_reconstructed_rl, 0.0)

# z_target = z_target*np.max(spprobe)/np.max(z_target)

# fit_gauss, fit_params = fit_n_gaussians_1d(
#     y_vals=om_grid,
#     z_vals=z_target,
#     n=n_comp,
#     return_params=True,
#     return_components=False,
#     use_weights=False,
# )

# probes_reconstructed = nfit_params_to_probes(fit_params)

# plt.figure()
# plt.plot(obs_Erange*hbar, z_target, label='RL target')
# plt.plot(obs_Erange*hbar, fit_gauss, label=f'{n_comp}-Gaussian fit')
# plt.plot(obs_Erange*hbar, spprobe, label='True spectrum')
# plt.xlabel('E - E0')
# plt.ylabel('Amplitude (normalized)')
# plt.title('n-Gaussian decomposition of RL reconstruction')
# plt.legend()
# plt.tight_layout()
# plt.show()


# correction = correcting_function_multi(OM_T,E,pulse_xuv,probes,dzeta=0.05)
# amplit_tot_FT_corrected = correction*amplit_tot_FT

# correction_rec = correcting_function_multi(OM_T,E,pulse_xuv,probes_reconstructed,dzeta=0.1)
# amplit_tot_FT_corrected_rec = correction_rec*amplit_tot_FT

# ###
# ### RESAMPLE AND ANALYZE
# ###

# rho_lo = 59.6
# rho_hi = 61.4
# rho_reconstructed, amplit_tot_FT_corrected_small, extent_small, idxs_small = resample(amplit_tot_FT_corrected,rho_hi,rho_lo,om_ref,E,OM_T,N_T)

# rho_lo = 59.6
# rho_hi = 61.4
# rho_reconstructed_rec, amplit_tot_FT_corrected_small_rec, extent_small, idxs_small = resample(amplit_tot_FT_corrected_rec,rho_hi,rho_lo,om_ref,E,OM_T,N_T)

# rho_reconstructed_uncor, amplit_tot_FT_corrected_small_uncor, extent_small, idxs_small = resample(amplit_tot_FT,rho_hi,rho_lo,om_ref,E,OM_T,N_T)

# plot_mat(rho_reconstructed_uncor,extent=[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',
#          mode='abs',saveloc='newims/uncorr.png',xlabel='Energy [eV]',ylabel='Energy [eV]',
#          title='Rho uncorrected for the probe spectrum')

# ###
# ### PLOT AND COMPARE TO THEORY
# ###

# E_range_new = np.linspace(rho_lo,rho_hi,N_E)

# E1,E2 = np.meshgrid(E_range_new,E_range_new)

# rho_synthetic_0 = np.exp(-1/2*((E1/hbar-om_xuv)**2/s_xuv**2 + (E2/hbar-om_xuv)**2/s_xuv**2))
# rho_synthetic = normalize_rho(rho_synthetic_0)


# plot_mat(rho_reconstructed,extent=[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',
#          mode='abs',saveloc='newims/corr_known.png',xlabel='Energy [eV]',ylabel='Energy [eV]',
#          title='Rho corrected (exact probe spectrum)')

# plot_mat(rho_reconstructed_rec,extent=[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',
#          mode='abs',saveloc='newims/corr_unknown.png',xlabel='Energy [eV]',ylabel='Energy [eV]',
#          title='Rho corrected (retrieved probe spectrum)')


# plot_mat(rho_reconstructed,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')
# plot_mat(rho_synthetic,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')
# plot_mat(np.minimum(np.abs((rho_synthetic-rho_reconstructed)/rho_synthetic),0.1),[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')

# plot_mat(rho_reconstructed_rec,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')
# plot_mat(rho_synthetic,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')
# plot_mat(np.minimum(np.abs((rho_synthetic-rho_reconstructed_rec)/rho_synthetic),0.1),[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')

# ######
# ######
# ######

# def _psd_unit_trace(A, eps=1e-12):
#     # Hermitian projection
#     H = 0.5 * (A + A.conj().T)
#     # Eigen-decomposition
#     w, V = np.linalg.eigh(H)
#     # Clip small negatives, rebuild PSD matrix
#     w_clipped = np.clip(w.real, 0.0, None)
#     H_psd = V @ (w_clipped[:, None] * V.conj().T)
#     # Normalize to unit trace
#     tr = float(np.trace(H_psd).real)
#     if tr <= eps:
#         n = H_psd.shape[0]
#         return np.eye(n) / float(n)
#     return H_psd / tr

# def _sqrt_psd(A):
#     # A should be Hermitian PSD
#     H = 0.5 * (A + A.conj().T)
#     w, V = np.linalg.eigh(H)
#     w = np.clip(w.real, 0.0, None)
#     return V @ (np.sqrt(w)[:, None] * V.conj().T)

# def density_fidelity(rho, sigma):
#     # Uhlmann fidelity: F(ρ,σ) = (Tr sqrt( sqrt(ρ) σ sqrt(ρ) ))^2
#     rho_p = _psd_unit_trace(rho)
#     sigma_p = _psd_unit_trace(sigma)
#     sqrt_rho = _sqrt_psd(rho_p)
#     M = sqrt_rho @ sigma_p @ sqrt_rho
#     M = 0.5 * (M + M.conj().T)
#     w, _ = np.linalg.eigh(M)
#     tr_sqrt = np.sum(np.sqrt(np.clip(w.real, 0.0, None))).real
#     F = float(tr_sqrt**2)
#     return float(np.clip(F, 0.0, 1.0))

# F_syn_rec = density_fidelity(normalize_rho(rho_synthetic), normalize_rho(rho_reconstructed))
# F_syn_rec_rec = density_fidelity(normalize_rho(rho_synthetic), normalize_rho(rho_reconstructed_rec))

# print(f"Fidelity(rho_synthetic, rho_reconstructed)      = {F_syn_rec:.6f}")
# print(f"Fidelity(rho_synthetic, rho_reconstructed_rec) = {F_syn_rec_rec:.6f}")