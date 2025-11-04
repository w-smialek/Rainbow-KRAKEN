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

def CFT(T_range, signal, use_window=True):
    """
    Continuous-time Fourier transform convention per column (fixed energy E):
        S(ω, E) = ∫ s(t, E) · e^{-i ω t} dt

    Discrete implementation with FFT for samples t_n = t0 + n·dt:
        S(ω_k, E) ≈ dt · e^{-i ω_k t0} · FFT_n{ s(t_n, E) }[k]

    If use_window is True, a Hann window is applied along time to reduce spectral leakage
    and we divide by its coherent gain so line amplitudes are preserved. If False, a
    rectangular window (no windowing) is used.

    Returns:
      - spec_shift: FFT along time (rows), shifted so ω runs from negative to positive
      - OM_T: 2D array of angular frequency (ω) matching spec_shift shape along rows
      - E_min, E_max: energy-axis bounds (ħ·ω) corresponding to the first/last row of spec_shift
    """
    # Validate input and infer sizes locally (avoid globals)
    t = np.asarray(T_range)
    if t.ndim != 1:
        raise ValueError("T_range must be a 1D array of time samples")
    NT_local = t.size
    if signal.shape[0] != NT_local:
        raise ValueError("signal shape[0] must match len(T_range)")
    NE_local = signal.shape[1]

    # Ensure uniform sampling
    if NT_local < 2:
        raise ValueError("T_range must contain at least two samples")
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=0.0):
        raise ValueError("T_range must be uniformly spaced")

    # Windowing
    if use_window:
        window = np.hanning(NT_local)[:, None]  # column vector
        coherent_gain = window.mean()           # for Hann, exactly 0.5
    else:
        window = np.ones((NT_local, 1))
        coherent_gain = 1.0

    windowed = (signal * window) / coherent_gain

    # Frequencies (unshifted) and corresponding angular frequencies
    freqs = np.fft.fftfreq(NT_local, d=dt)   # cycles per unit time
    omega = 2 * np.pi * freqs                # rad / unit time

    # FFT along time axis with dt scaling to approximate the continuous integral
    spec = np.fft.fft(windowed, axis=0) * dt

    # Phase correction for non-zero time origin t0 = T_range[0]
    t0 = t[0]
    phase = np.exp(-1j * omega[:, None] * t0)
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

    # Sig_cc_cubic_mesh = 1/2*(Sig_cc_cubic_mesh + np.conjugate(Sig_cc_cubic_mesh.T))

    Sig_cc_cubic_mesh = Sig_cc_cubic_mesh/np.sum(np.diag(np.abs(Sig_cc_cubic_mesh)))

    return Sig_cc_cubic_mesh, small_sig, extent, [idy_min,idy_max,idx_min,idx_max]

def plot_mat(mat,extent,cmap='viridis',mode='abs'):

    if mode == 'abs':
        plt.figure()
        im = plt.imshow(np.abs(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('|M|')
        plt.colorbar(im)
        plt.show()

    elif mode == 'phase':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        im0 = axes[0].imshow(np.abs(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[0].set_title('|M|')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(np.angle(mat), extent=extent, origin='lower', aspect='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('arg(M)')
        axes[1].set_xlabel('x')
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()

    elif mode == 'reim':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        im0 = axes[0].imshow(np.real(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[0].set_title('Re{M}')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(np.imag(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[1].set_title('Im{M}')
        axes[1].set_xlabel('x')
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()
    return

def normalize_rho(rho):
    return rho/np.sum(np.diag(np.abs(rho)))

def _sum3_gauss1d(y, A1, mu1, s1, A2, mu2, s2, A3, mu3, s3):
    """Sum of three 1D Gaussians."""
    g1 = A1 * np.exp(-0.5 * ((y - mu1) / s1) ** 2)
    g2 = A2 * np.exp(-0.5 * ((y - mu2) / s2) ** 2)
    g3 = A3 * np.exp(-0.5 * ((y - mu3) / s3) ** 2)
    return g1 + g2 + g3

def fit_three_gaussians_1d(y_vals, z_vals, n_center_rows=3, center_bounds_frac=0.25, width_bounds=(0.05, 2.0), center_index=None):
    """Fit a sum of three 1D Gaussians to a single column z(y).

    - Fit is performed excluding n_center_rows around the central y-index (masking the spike region).
    - Coordinates are recentered so mu in initial guesses is near 0, with bounds to avoid drift.
    - Width bounds are fractions of the total y-span for robustness.

    Returns the fitted curve evaluated on the full y grid.
    """
    y_vals = np.asarray(y_vals).ravel()
    z_vals = np.asarray(z_vals).ravel()
    ny = y_vals.size
    if z_vals.size != ny:
        raise ValueError("z_vals must have same length as y_vals")

    # Recenter y so that 0 aligns with the Dirac-spike row
    c_row = ny // 2 if center_index is None else int(center_index)
    c_row = max(0, min(ny - 1, c_row))
    y0 = y_vals[c_row]
    yc = y_vals - y0
    span_y = max(np.ptp(yc), 1e-9)

    # Mask central rows
    m = max(1, int(n_center_rows))
    if m % 2 == 0:
        m += 1
    half = m // 2
    c_row = ny // 2 if center_index is None else int(center_index)
    c_row = max(0, min(ny - 1, c_row))
    lo = max(0, c_row - half)
    hi = min(ny, c_row + half + 1)
    mask = np.ones(ny, dtype=bool)
    mask[lo:hi] = False

    y_fit = yc[mask]
    z_fit = z_vals[mask]

    # Bounds and initial guesses
    min_s = float(width_bounds[0]) * span_y
    max_s = float(width_bounds[1]) * span_y
    c_bound = float(center_bounds_frac) * span_y

    A_scale = float(np.median(np.abs(z_fit)))
    A_scale = max(A_scale, 1e-12)

    p0 = [
        A_scale/2, 0.0, 0.6*span_y,
        A_scale/3, 0.0, 0.3*span_y,
        A_scale/6, 0.0, 1.2*span_y,
    ]
    lb = [0.0, -c_bound, min_s, 0.0, -c_bound, min_s, 0.0, -c_bound, min_s]
    ub = [np.inf,  c_bound, max_s, np.inf,  c_bound, max_s, np.inf,  c_bound, max_s]

    try:
        popt, _ = curve_fit(
            _sum3_gauss1d,
            y_fit,
            z_fit,
            p0=p0,
            bounds=(lb, ub),
            maxfev=40000,
        )
        fit_full = _sum3_gauss1d(yc, *popt)
    except Exception:
        fit_full = np.zeros_like(z_vals)

    return fit_full

def _gauss1d_centered(y, A, s):
    """Single Gaussian centered at 0: A * exp(-0.5 * (y/s)^2)."""
    return A * np.exp(-0.5 * (y / s) ** 2)

def fit_single_gaussian_centered_1d(y_vals, z_vals, n_center_rows=3, width_bounds=(0.05, 2.0), center_index=None):
    """Fit a single Gaussian per column, centered at the middle row (fixed center).

    - Recenter y to zero so the Gaussian center is at 0 by construction.
    - Omit n_center_rows around y-center during fitting to avoid the spike.
    - Fit parameters: amplitude A >= 0, width s in [width_bounds[0], width_bounds[1]] * span_y.

    Returns the fitted curve evaluated on the full y grid.
    """
    y_vals = np.asarray(y_vals).ravel()
    z_vals = np.asarray(z_vals).ravel()
    ny = y_vals.size
    if z_vals.size != ny:
        raise ValueError("z_vals must have same length as y_vals")

    # Recenter y so 0 is exactly at the Dirac-spike row
    c_row = ny // 2 if center_index is None else int(center_index)
    c_row = max(0, min(ny - 1, c_row))
    y0 = y_vals[c_row]
    yc = y_vals - y0
    span_y = max(np.ptp(yc), 1e-9)

    # Mask out the central rows during fitting
    m = max(1, int(n_center_rows))
    if m % 2 == 0:
        m += 1
    half = m // 2
    c_row = ny // 2 if center_index is None else int(center_index)
    c_row = max(0, min(ny - 1, c_row))
    lo = max(0, c_row - half)
    hi = min(ny, c_row + half + 1)
    mask = np.ones(ny, dtype=bool)
    mask[lo:hi] = False

    y_fit = yc[mask]
    z_fit = z_vals[mask]

    # Initial guesses and bounds
    min_s = float(width_bounds[0]) * span_y
    max_s = float(width_bounds[1]) * span_y
    A0 = float(np.median(np.abs(z_fit)))
    A0 = max(A0, 1e-12)
    s0 = 0.6 * span_y

    try:
        popt, _ = curve_fit(
            _gauss1d_centered,
            y_fit,
            z_fit,
            p0=[A0, s0],
            bounds=([0.0, min_s], [np.inf, max_s]),
            maxfev=20000,
        )
        fit_full = _gauss1d_centered(yc, *popt)
    except Exception:
        fit_full = np.zeros_like(z_vals)

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

def plot_spectra(gausses, n_points=1000, sigma_span=6.0):
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
    plt.xlabel('E')
    plt.ylabel('Amplitude')
    plt.tight_layout()
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

def Amplitude(xuvs,refprobes,E,E_spinorbit=0):
    n_t, n_e = E.shape
    amplit_tot = np.zeros((n_t,n_e)).astype(complex)

    xuvs_mod = []
    for xuv in xuvs:
        tau_i,A_i,om0_i,s_i,phi_i = xuv
        xuvs_mod.append((tau_i,A_i,om0_i-E_spinorbit/hbar,s_i,phi_i))

    for xuv in xuvs_mod:
        for rp in refprobes:
            amplit_tot += Amplitude_ij(xuv,rp,E)
            amplit_tot += Amplitude_ij(rp,xuv,E)
    return amplit_tot

###
### FIELD PARAMETERS
###

E_lo = 60.5
E_hi = 63.5
T_reach = 100

N_E = 1700
N_T = 1700

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

A_probe2 = 0.1
om_probe2 = 1.05/hbar
s_probe2 = 0.05/hbar
pulse_probe2 = (T,A_probe2,om_probe2,s_probe2,0)

A_probe3 = 0.3
om_probe3 = 2.10/hbar
s_probe3 = 0.03/hbar
pulse_probe3 = (T,A_probe3,om_probe3,s_probe3,0)

A_ref = 1
om_ref = 1.55/hbar
s_ref = 0.005/hbar
pulse_ref = (0*T,A_ref,om_ref,s_ref,0)

refs = (pulse_ref,)
probes = (pulse_probe,pulse_probe2,pulse_probe3)
xuvs = (pulse_xuv,)#,pulse_xuv2,pulse_xuv3]
refprobes = tuple(list(refs) + list(probes))

plot_spectra(probes)

###
### GENERATE SIGNAL
###

amplit_tot_0 = Amplitude(xuvs,refprobes,E)

# Transition dipole element (square modulus summed over ionization OAM channels) is approximated as T_i(E) = a_i*(E-E_0)

a_dipole_0 = 0.00

SNR = 30

signal_clean = (1 + a_dipole_0*(E-om_xuv*hbar)) * np.abs(amplit_tot_0)**2

if SNR is None or SNR <= 0:
    signal = signal_clean.copy()
else:
    sig_rms = np.sqrt(np.mean(signal_clean**2))
    noise_rms = sig_rms / float(SNR)
    noise = noise_rms * np.random.normal(size=signal_clean.shape)
    signal = signal_clean + noise

amplit_tot_FT, OM_T, em_lo, em_hi = CFT(T_range,signal,use_window=False)

plot_mat(signal,[E_lo,E_hi,-T_reach,T_reach],cmap='plasma',mode='phase')
# plot_mat(np.minimum(np.abs(amplit_tot_FT),10),[E_lo,E_hi,em_lo,em_hi],cmap='plasma',mode='phase')

###
### DETRENDING - SEPARATING PROBE AND REF ZERO FREQ COMPONENT 
###

mid_emrange_lo = 0.47
mid_emrange_hi = 0.53

# --- Fit and subtract slowly varying background in each column ---
# Build the y-axis (energy units ħ·ω) for the selected mid band
i0 = floor(N_T*mid_emrange_lo)
i1 = floor(N_T*mid_emrange_hi)
em_axis_full = hbar * OM_T[:, 0]
em_axis_mid = em_axis_full[i0:i1]

amplit_tot_FT_mid = amplit_tot_FT[i0:i1,:]
plot_mat(np.clip(np.abs(amplit_tot_FT_mid),0,5),[E_lo,E_hi,em_axis_mid[0],em_axis_mid[-1]],cmap='plasma',mode='phase')

# Prepare baseline via column-wise 1D single-Gaussian fits (omit n central rows in fitting)
abs_mid = np.abs(amplit_tot_FT_mid)

# x-axis is energy (columns); y-axis is ħ·ω (rows)
x_axis = E[0, :]
y_axis = em_axis_mid

# Omit n central rows when fitting (Dirac-like spike)
n_central_rows = 3
ny, nx = abs_mid.shape

# Detect the spike row on the entire (un-sliced) spectrum and map into the mid-band slice
abs_full = np.abs(amplit_tot_FT)
spike_row_full = int(np.argmax(np.sum(abs_full, axis=1)))
spike_row_mid = spike_row_full - i0
if spike_row_mid < 0 or spike_row_mid >= ny:
    spike_row_mid = ny // 2  # fallback if spike outside selected band

# Fit per column outside the central rows and evaluate on full grid
fit_cols = np.zeros_like(abs_mid)
for j in range(nx):
    fit_cols[:, j] = fit_single_gaussian_centered_1d(y_axis, abs_mid[:, j], n_center_rows=n_central_rows, center_index=spike_row_mid)

# Compute spike as residual but restrict it to central rows only
m = max(1, int(n_central_rows))
if m % 2 == 0:
    m += 1
half = m // 2
c_row = spike_row_mid
lo = max(0, c_row - half)
hi = min(ny, c_row + half + 1)
spike_mask = np.zeros_like(abs_mid, dtype=bool)
spike_mask[lo:hi, :] = True

residual = abs_mid - fit_cols
residual_pos = np.maximum(residual, 0.0)

# # Build threshold from the row with the maximum average residual and apply globally
# threshold_frac = 0.1  # fraction of strongest-row average to keep
# if ny > 0:
#     row_avgs = np.mean(residual_pos, axis=1)
#     base_mean = float(row_avgs[np.argmax(row_avgs)])
# else:
#     base_mean = 0.0
# thr = threshold_frac * base_mean
# above_thr_mask = residual_pos >= thr

apply_mask = spike_mask# & above_thr_mask


# Spike-only map for diagnostics: positive residuals within central rows and above threshold
spike_only = np.where(apply_mask, residual_pos, 0.0)

# Baseline uses the fitted 1D Gaussian inside spike areas; elsewhere keep original magnitude
baseline_mag = abs_mid.copy()
baseline_mag[apply_mask] = fit_cols[apply_mask]

# Reconstruct complex result keeping phase but with new baseline magnitude
eps = 1e-12
scale = np.divide(np.clip(baseline_mag, 0.0, None), abs_mid + eps)
amplit_tot_FT_mid_detrended = amplit_tot_FT_mid * scale



# Plot diagnostics
extent_mid = [E_lo, E_hi, float(em_axis_mid[0]), float(em_axis_mid[-1])]


###
### EXTRACT PROBE SPECTRUM
###

true_pramp = Amplitude(xuvs,probes,E)
true_prsig = np.abs(true_pramp)**2

true_prsigft, OM_T, em_lo, em_hi = CFT(T_range,true_prsig,use_window=False)
true_prsigftmid = true_prsigft[i0:i1]

plot_mat(fit_cols, extent_mid, cmap='plasma', mode='phase')
plot_mat(true_prsigftmid, extent_mid, cmap='plasma', mode='phase')
plot_mat(fit_cols-true_prsigftmid, extent_mid, cmap='plasma', mode='phase')



plot_mat(baseline_mag, extent_mid, cmap='plasma', mode='phase')
plot_mat(amplit_tot_FT_mid_detrended, extent_mid, cmap='plasma', mode='phase')
plot_mat(spike_only, extent_mid, cmap='plasma', mode='phase')

# Find the row containing the global maximum magnitude and extract it
abs_mid_det = np.abs(amplit_tot_FT_mid_detrended)
max_row_idx, _ = np.unravel_index(np.argmax(abs_mid_det), abs_mid_det.shape)
delta_ofEt = amplit_tot_FT_mid_detrended[max_row_idx, :]

plt.plot(delta_ofEt)
plt.show()


###
### CORRECTION ANALYTICALLY
###



correction = correcting_function_multi(OM_T,E,pulse_xuv,probes,dzeta=0.02)
amplit_tot_FT_corrected = correction*amplit_tot_FT

###
### RESAMPLE AND ANALYZE
###

rho_lo = 59.6
rho_hi = 61.4
rho_reconstructed, amplit_tot_FT_corrected_small, extent_small, idxs_small = resample(amplit_tot_FT_corrected,rho_hi,rho_lo,om_ref,E,OM_T,N_T)

# plot_mat(amplit_tot_FT_corrected_small,extent_small,cmap='plasma',mode='phase')

###
### PLOT AND COMPARE TO THEORY
###

E_range_new = np.linspace(rho_lo,rho_hi,N_E)

E1,E2 = np.meshgrid(E_range_new,E_range_new)

rho_synthetic_0 = np.exp(-1/2*((E1/hbar-om_xuv)**2/s_xuv**2 + (E2/hbar-om_xuv)**2/s_xuv**2))

rho_synthetic = normalize_rho(rho_synthetic_0)

plot_mat(rho_reconstructed,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')
plot_mat(rho_synthetic,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')
plot_mat(np.minimum(np.abs((rho_synthetic-rho_reconstructed)/rho_synthetic),0.1),[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')