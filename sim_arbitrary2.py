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

def _sum3_gauss2d(XY,
                  A1, x01, y01, sx1, sy1,
                  A2, x02, y02, sx2, sy2,
                  A3, x03, y03, sx3, sy3):
    """Sum of three axis-aligned 2D Gaussians with independent centers.
    XY should be a 2xN array where XY[0] = x coords, XY[1] = y coords.
    Each term: A*exp(-1/2*((x-x0)/sx)^2) * exp(-1/2*((y-y0)/sy)^2).
    """
    x = XY[0]
    y = XY[1]
    g1 = A1 * np.exp(-0.5 * (((x - x01) / sx1) ** 2 + ((y - y01) / sy1) ** 2))
    g2 = A2 * np.exp(-0.5 * (((x - x02) / sx2) ** 2 + ((y - y02) / sy2) ** 2))
    g3 = A3 * np.exp(-0.5 * (((x - x03) / sx3) ** 2 + ((y - y03) / sy3) ** 2))
    return g1 + g2 + g3

def fit_three_gaussians_2d(x_vals, y_vals, Z, n_center_rows=3):
    """Fit sum of three 2D Gaussians with centers:
        A exp(-1/2*((x-x0)/sx)^2) exp(-1/2*((y-y0)/sy)^2)
    to the magnitude Z(y, x).

    Implementation details:
      - Recenter coordinates by subtracting midpoints of x_vals, y_vals so initial centers ~ 0.
      - Omit n_center_rows around the central y-index to avoid the Dirac-like spike.
      - Use bounded non-linear least squares with positive amplitudes and widths,
        and modest center bounds to avoid overfitting noise.

    Returns: baseline evaluated on the full (y,x) grid.
    """
    x_vals = np.asarray(x_vals).ravel()
    y_vals = np.asarray(y_vals).ravel()
    Z = np.asarray(Z)
    ny, nx = Z.shape
    if nx != x_vals.size or ny != y_vals.size:
        raise ValueError("Z shape must be (len(y_vals), len(x_vals))")

    # Center coordinates at 0
    x0 = 0.5 * (x_vals.min() + x_vals.max())
    y0 = 0.5 * (y_vals.min() + y_vals.max())
    Xc = x_vals - x0
    Yc = y_vals - y0

    # Build full coordinate grid (rows=y, cols=x)
    Xg, Yg = np.meshgrid(Xc, Yc, indexing='xy')

    # Mask out n_center_rows around the central row
    m = max(1, int(n_center_rows))
    if m % 2 == 0:
        m += 1
    half = m // 2
    c_row = ny // 2
    lo = max(0, c_row - half)
    hi = min(ny, c_row + half + 1)
    mask = np.ones((ny, nx), dtype=bool)
    mask[lo:hi, :] = False

    x_fit = Xg[mask]
    y_fit = Yg[mask]
    z_fit = Z[mask]

    # Initial guesses and bounds
    span_x = max(np.ptp(Xc), 1e-6)
    span_y = max(np.ptp(Yc), 1e-6)
    min_sx = 0.05 * span_x
    max_sx = 2.00 * span_x
    min_sy = 0.05 * span_y
    max_sy = 2.00 * span_y

    # Allow centers to move within a modest fraction of the span
    cx_bound = 0.25 * span_x
    cy_bound = 0.25 * span_y

    A_scale = float(np.median(z_fit))
    A_scale = max(A_scale, 1e-12)

    # Initial guesses: small centers near 0, a range of widths and amplitudes
    p0 = [
        A_scale/2,  0.0,  0.0,  0.6*span_x, 0.6*span_y,
        A_scale/3,  0.0,  0.0,  0.3*span_x, 0.3*span_y,
        A_scale/6,  0.0,  0.0,  1.2*span_x, 1.2*span_y,
    ]
    lb = [
        0.0, -cx_bound, -cy_bound, min_sx, min_sy,
        0.0, -cx_bound, -cy_bound, min_sx, min_sy,
        0.0, -cx_bound, -cy_bound, min_sx, min_sy,
    ]
    ub = [
        np.inf, cx_bound, cy_bound, max_sx, max_sy,
        np.inf, cx_bound, cy_bound, max_sx, max_sy,
        np.inf, cx_bound, cy_bound, max_sx, max_sy,
    ]

    try:
        popt, _ = curve_fit(
            _sum3_gauss2d,
            np.vstack([x_fit, y_fit]),
            z_fit,
            p0=p0,
            bounds=(lb, ub),
            maxfev=80000,
        )
        baseline = _sum3_gauss2d(np.vstack([Xg.ravel(), Yg.ravel()]), *popt).reshape(ny, nx)
    except Exception:
        # Fallback: zero baseline
        baseline = np.zeros_like(Z)

    return baseline

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
    return A/(2*s)*np.exp(-(om-om0)**2/(2*s**2))  # Positive-freq part

def sp_tot(gausses,om):
    retval = 0
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

N_E = 1200
N_T = 1200

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
om_probe2 = 1.45/hbar
s_probe2 = 0.05/hbar
pulse_probe2 = (T,A_probe2,om_probe2,s_probe2,0)

A_probe3 = 0.05
om_probe3 = 1.80/hbar
s_probe3 = 0.03/hbar
pulse_probe3 = (T,A_probe3,om_probe3,s_probe3,0)

A_ref = 1
om_ref = 1.55/hbar
s_ref = 0.005/hbar
pulse_ref = (0*T,A_ref,om_ref,s_ref,0)

refs = [pulse_ref]
probes = [pulse_probe,pulse_probe2,pulse_probe3]
xuvs = [pulse_xuv]#,pulse_xuv2,pulse_xuv3]
refprobes = refs + probes

# plot_spectra(probes)

###
### GENERATE SIGNAL
###

E_spinorbit = 0.18
# E_spinorbit = 0.0

amplit_tot_0 = Amplitude(xuvs,refprobes,E)
amplit_tot_s = Amplitude(xuvs,refprobes,E,E_spinorbit)

# Transition dipole element (square modulus summed over ionization OAM channels) is approximated as T_i(E) = a_i*(E-E_0)

a_dipole = 0.2

a_dipole_0 = 0.4
a_dipole_s = 0.3

# signal = (1 + a_dipole*(E - om_xuv*hbar)) * ( (2/3) * np.abs(amplit_tot_0)**2 + (1/3) * np.abs(amplit_tot_s)**2 )
# desired linear SNR = signal_rms / noise_rms (set externally if you like, default 50)
SNR = 30

signal_clean = (2/3 + a_dipole_0*(E-om_xuv*hbar)) * np.abs(amplit_tot_0)**2 + (1/3 + a_dipole_s*(E-om_xuv*hbar)) * np.abs(amplit_tot_s)**2

if SNR is None or SNR <= 0:
    signal = signal_clean.copy()
else:
    sig_rms = np.sqrt(np.mean(signal_clean**2))
    noise_rms = sig_rms / float(SNR)
    noise = noise_rms * np.random.normal(size=signal_clean.shape)
    signal = signal_clean + noise

amplit_tot_FT, OM_T, em_lo, em_hi = CFT(T_range,signal,use_window=False)

plot_mat(signal,[E_lo,E_hi,-T_reach,T_reach],cmap='plasma',mode='phase')
plot_mat(np.minimum(np.abs(amplit_tot_FT),10),[E_lo,E_hi,em_lo,em_hi],cmap='plasma',mode='phase')

###
### 
###

mid_emrange_lo = 0.47
mid_emrange_hi = 0.53

# --- Fit and subtract slowly varying background in each column ---
# Build the y-axis (energy units ħ·ω) for the selected mid band
i0 = floor(N_T*mid_emrange_lo)
i1 = floor(N_T*mid_emrange_hi)
em_axis_full = hbar * OM_T[:, 0]
em_axis_mid = em_axis_full[i0:i1]

amplit_tot_FT_mid = amplit_tot_FT[floor(N_T*mid_emrange_lo):floor(N_T*mid_emrange_hi),:]
plot_mat(amplit_tot_FT_mid,[E_lo,E_hi,em_axis_mid[0],em_axis_mid[-1]],cmap='plasma',mode='phase')

# Prepare baseline and corrected arrays using 2D 3-Gaussian fit
abs_mid = np.abs(amplit_tot_FT_mid)

# x-axis is energy (columns); y-axis is ħ·ω (rows)
x_axis = E[0, :]
y_axis = em_axis_mid

# Omit n central rows when fitting (Dirac-like spike)
n_central_rows = 5
baseline = fit_three_gaussians_2d(x_axis, y_axis, abs_mid, n_center_rows=n_central_rows)

# Subtract baseline on magnitudes and reconstruct complex result keeping phase
eps = 1e-12
mag_corr = np.clip(abs_mid - baseline, 0.0, None)
scale = np.divide(mag_corr, abs_mid + eps)

amplit_tot_FT_mid_detrended = amplit_tot_FT_mid * scale

# Plot diagnostics
extent_mid = [E_lo, E_hi, float(em_axis_mid[0]), float(em_axis_mid[-1])]
plot_mat(baseline, extent_mid, cmap='plasma', mode='phase')
plot_mat(amplit_tot_FT_mid_detrended, extent_mid, cmap='plasma', mode='phase')

# Find the row containing the global maximum magnitude and extract it
abs_mid_det = np.abs(amplit_tot_FT_mid_detrended)
max_row_idx, _ = np.unravel_index(np.argmax(abs_mid_det), abs_mid_det.shape)
delta_ofEt = amplit_tot_FT_mid_detrended[max_row_idx, :]

###
### Extract dipole element variation
###

xuv_0 = np.abs(sp_tot(xuvs,E_range/hbar-om_ref))**2
xuv_s = np.abs(sp_tot(xuvs,E_range/hbar-om_ref + E_spinorbit/hbar))**2
xuv_baseline = 2/3 * xuv_0 + 1/3 * xuv_s

plt.plot(xuv_baseline)
plt.plot(xuv_0)
plt.show()



# Raw deconvolution (xuv_baseline = xuv_0 * k) via FFT with Tikhonov regularization
dE = E_range[1] - E_range[0]
N = xuv_0.size
L = 2*N - 1  # linear convolution length

X0 = np.fft.fft(xuv_0, n=L)
XB = np.fft.fft(delta_ofEt, n=L)
lam = 1e-8 * np.max(np.abs(X0)**2) + 1e-30
Kf = XB * np.conj(X0) / (np.abs(X0)**2 + lam)
k_raw = np.fft.ifft(Kf).real  # expected: ~ (2/3)δ[0] + (1/3)δ[+shift]

plt.plot(np.linspace(0,L*dE,L),k_raw)
plt.show()









plt.plot(3000*np.abs(delta_ofEt)/np.sum(np.abs(delta_ofEt)))
plt.plot(3000*xuv_0/np.sum(xuv_0))
plt.plot(np.clip(np.abs(delta_ofEt/xuv_0),0,100))

plt.plot(3000*xuv_s/np.sum(xuv_s))
plt.plot(np.clip(np.abs(delta_ofEt/xuv_s),0,100))
plt.show()

N = xuv_0.size
denoms = np.vstack([np.roll(xuv_0, k) for k in range(N)])
with np.errstate(divide='ignore', invalid='ignore'):
    ar = np.clip(np.abs(delta_ofEt / denoms), 0, 100)
ar = np.vstack([ar, ar])

plot_mat(ar,[1,2,1,2],cmap='plasma',mode='phase')

# Smooth, clip to [0, 100], then compute Laplacian

# Gaussian smoothing
ar_sm = gaussian_filter(ar, sigma=1.0)

# Clip values
ar_sm = np.clip(ar_sm, 0.0, 100.0)

# Laplacian of the smoothed, clipped array
ar = laplace(ar_sm, mode='reflect')

ar = np.clip(ar/ar_sm,-0.008,0.008)

plot_mat(ar,[(E_lo-E_hi)/2,(E_hi-E_lo)/2,E_lo-E_hi,E_hi-E_lo,],cmap='plasma',mode='phase')

# def fitdelta_so(E,dE_so,dE_0,a_dipole_0,b_dipole_0,a_dipole_s,b_dipole_s):
#     return (b_dipole_0 + a_dipole_0*(E-om_xuv*hbar))*np.abs(sp_tot(xuvs,E/hbar-om_ref-dE_0/hbar))**2 + (b_dipole_s + a_dipole_s*(E-om_xuv*hbar))*np.abs(sp_tot(xuvs,E/hbar - om_ref - dE_so/hbar))**2

# # Fit |delta_ofEt| to fitdelta_so(E, dE_so, a0, b0, aS, bS)
# xdata = E_range

# sc = np.sum(np.abs(delta_ofEt))
# ydata = np.abs(delta_ofEt)/sc

# # Initial guesses and bounds
# p0 = [-0.45,0.0 , 0.0, 0.7, 0.0, 0.3]  # [dE_so, a_dipole_0, b_dipole_0, a_dipole_s, b_dipole_s]

# popt, pcov = curve_fit(
#     fitdelta_so,
#     xdata,
#     ydata,
#     p0=p0,
# )
# perr = np.sqrt(np.diag(pcov))

# # Fitted curve
# yfit = fitdelta_so(xdata, *popt)

# print("Fitted parameters:")
# print(f"dE_so = {popt[0]:.6f} ± {perr[0]:.6f} eV")
# print(f"dE_0 = {popt[1]:.6f} ± {perr[1]:.6f} eV")
# print(f"a_dipole_0 = {sc*popt[2]:.6f} ± {sc*perr[2]:.6f}")
# print(f"b_dipole_0 = {sc*popt[3]:.6f} ± {sc*perr[3]:.6f}")
# print(f"a_dipole_s = {sc*popt[4]:.6f} ± {sc*perr[4]:.6f}")
# print(f"b_dipole_s = {sc*popt[5]:.6f} ± {sc*perr[5]:.6f}")

# # Plot data vs fit
# plt.figure()
# plt.plot(xdata, np.abs(delta_ofEt), label='|delta_ofEt|', alpha=0.7)
# plt.plot(xdata, sc*yfit, label='fit', lw=2)
# plt.xlabel('E (eV)')
# plt.ylabel('Amplitude (arb.)')
# plt.legend()
# plt.tight_layout()
# plt.show()

###
### CORRECTION ANALYTICALLY
###

correction = correcting_function_multi(OM_T,E,pulse_xuv,probes,dzeta=0.05)
amplit_tot_FT_corrected = correction*amplit_tot_FT

###
### RESAMPLE AND ANALYZE

rho_lo = 59.6
rho_hi = 61.4
rho_reconstructed, amplit_tot_FT_corrected_small, extent_small, idxs_small = resample(amplit_tot_FT_corrected,rho_hi,rho_lo,om_ref,E,OM_T,N_T)

# plot_mat(amplit_tot_FT_corrected_small,extent_small,cmap='plasma',mode='phase')

###
### Modulating function check
###

# mod_arr = modulating_function_multi(OM_T,E,pulse_xuv,probes) * np.exp(-1/2*((om_ref+om_xuv-E/hbar)/(s_xuv))**2)

# idy_min, idy_max, idx_min, idx_max = idxs
# mod_arr = mod_arr[idy_min:idy_max,idx_min:idx_max]

# small_norm = np.abs(small)/np.sum(np.abs(small))
# mod_arr_norm = np.abs(mod_arr)/np.sum(np.abs(mod_arr))

# plot_mat(small_norm,extent_small,cmap='plasma',mode='phase')
# plot_mat(mod_arr_norm,extent_small,cmap='plasma',mode='phase')

# plot_mat(small_norm-mod_arr_norm,extent_small,cmap='plasma',mode='phase')

# plt.plot(small_norm[:,100])
# plt.plot(mod_arr_norm[:,100])
# plt.show()

###
### PLOT AND COMPARE TO THEORY
###

E_range_new = np.linspace(rho_lo,rho_hi,N_E)

E1,E2 = np.meshgrid(E_range_new,E_range_new)

rho_synthetic_0 = np.exp(-1/2*((E1/hbar-om_xuv)**2/s_xuv**2 + (E2/hbar-om_xuv)**2/s_xuv**2))

rho_synthetic_s = np.exp(-1/2*((E1/hbar-om_xuv+E_spinorbit/hbar)**2/s_xuv**2 + (E2/hbar-om_xuv+E_spinorbit/hbar)**2/s_xuv**2))

rho_synthetic = normalize_rho(2/3*rho_synthetic_0 + 1/3*rho_synthetic_s)

plot_mat(rho_synthetic,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')
plot_mat(rho_reconstructed,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')
plot_mat(np.minimum(np.abs((rho_synthetic-rho_reconstructed)/rho_synthetic),0.1),[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')