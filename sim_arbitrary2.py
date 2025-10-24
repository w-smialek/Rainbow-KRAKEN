import numpy as np
from numpy import pi
from scipy.special import erf
from scipy.special import wofz
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate
from scipy import integrate

# Reduced Planck constant in eV*fs (approx CODATA): ħ = 6.582119569e-16 eV·s
hbar = 6.582119569e-1

T_eg = 1 # preliminary simplification that it is const, in atomic units it can be roughly 1-0.01 for 0-100eV

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
    return prefactor*T_eg*wofz(wofz_arg)

def modulating_function(om_t,ene_Eg,pulse_params_x,pulse_params_p,big_sigma):
    tau_x,A_x,om0_x,s_x,phi_x = pulse_params_x
    tau_p,A_p,om0_p,s_p,phi_p = pulse_params_p
    s_p = s_p + big_sigma
    s = np.sqrt(s_x**2 + s_p**2)
    s_t = np.sqrt(s_x**(-2) + s_p**(-2))

    delta_p = om0_x + om0_p - ene_Eg/hbar
    delta_E = om0_x - ene_Eg/hbar - s_x**2/s**2 * delta_p

    return np.exp(-1/2*((delta_p/s)**2 + (s_t * (om_t + delta_E))**2))

def correcting_function(om_t,ene_Eg,pulse_params_x,pulse_params_p,dzeta=1e-3,big_sigma=1000):
    mod_plus = modulating_function(om_t,ene_Eg,pulse_params_x,pulse_params_p,0)
    err_area_plus = np.where(mod_plus<dzeta)
    mod_plus[err_area_plus] = 1
    mod2_plus = modulating_function(om_t,ene_Eg,pulse_params_x,pulse_params_p,big_sigma)

    correction_plus = om_t*mod2_plus/mod_plus
    correction_plus[err_area_plus] = 0
    return correction_plus

def modulating_function_multi(om_t,ene_Eg,pulse_params_x,probes,big_sigma):
    tau_x,A_x,om0_x,s_x,phi_x = pulse_params_x

    retval = 0
    A_tot = 0

    for probe in probes:
        tau_p,A_p,om0_p,s_p,phi_p = probe

        s_pp = s_p + big_sigma
        s = np.sqrt(s_x**2 + s_pp**2)
        s_t = np.sqrt(s_x**(-2) + s_pp**(-2))

        delta_p = om0_x + om0_p - ene_Eg/hbar
        delta_E = om0_x - ene_Eg/hbar - s_x**2/s**2 * delta_p

        retval += A_p/s_p*np.exp(-1/2*((delta_p/s)**2 + (s_t * (om_t + delta_E))**2))

        # plot_mat(retval,[1,2,1,2],cmap='plasma',mode='abs')

    return retval

def correcting_function_multi(om_t,ene_Eg,pulse_params_x,probes,dzeta=1e-3,big_sigma=1000):
    mod_plus = modulating_function_multi(om_t,ene_Eg,pulse_params_x,probes,0)
    err_area_plus = np.where(mod_plus<dzeta)
    mod_plus[err_area_plus] = 1
    mod2_plus = modulating_function_multi(om_t,ene_Eg,pulse_params_x,probes,big_sigma)

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

def CFT(T_range, signal):
    """
    Continuous-time Fourier transform convention per column (fixed energy E):
        S(ω, E) = ∫ s(t, E) · e^{-i ω t} dt

    Discrete implementation with FFT for samples t_n = t0 + n·dt:
        S(ω_k, E) ≈ dt · e^{-i ω_k t0} · FFT_n{ s(t_n, E) }[k]

    We also apply a Hann window along time to reduce spectral leakage and
    divide by its coherent gain so line amplitudes are preserved.
    """
    # apply Hann window along time (rows) for each column (energy)
    dt = T_range[1] - T_range[0]
    window = np.hanning(N_T)[:, None]  # column vector to multiply each column

    # Coherent gain (mean of window) for Hann is 0.5; compute generically
    coherent_gain = (window.mean())
    windowed = (signal * window) / coherent_gain

    # frequencies (unshifted) and corresponding angular frequencies
    freqs = np.fft.fftfreq(N_T, d=dt)        # cycles per unit time
    omega = 2 * np.pi * freqs                # rad / unit time

    # FFT along time axis with dt scaling to approximate the continuous integral
    spec = np.fft.fft(windowed, axis=0) * dt

    # Phase correction for non-zero time origin t0 = T_range[0]
    t0 = T_range[0]
    phase = np.exp(-1j * omega[:, None] * t0)
    spec *= phase

    # Shift spectrum and build energy axis (ħ·ω)
    spec_shift = np.fft.fftshift(spec, axes=0)
    energy_axis = hbar * omega               # energy axis in same units as hbar
    energy_axis_shift = np.fft.fftshift(energy_axis)

    OM_T = energy_axis_shift/hbar
    OM_T = np.tile(OM_T,(N_E,1)).T

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

    Ef_range = np.linspace(new_Ef_min,new_Ef_max,idx_max-idx_min)
    omt_range = np.linspace(new_omt_min,new_omt_max,idy_max-idy_min)

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

    return Sig_cc_cubic_mesh, small_sig, extent

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

E_lo = 60.5
E_hi = 63.5
T_reach = 100

N_E = 700
N_T = 700

E_range = np.linspace(E_lo,E_hi,N_E)
T_range = np.linspace(-T_reach,T_reach,N_T)
E, T = np.meshgrid(E_range,T_range)

A_xuv = 1
om_xuv = 60.65/hbar
s_xuv = 0.15/hbar
pulse_xuv = (0*T,A_xuv,om_xuv,s_xuv,0)

# A_xuv2 = 0.3
# om_xuv2 = 60.75/hbar
# s_xuv2 = 0.05/hbar
# pulse_xuv2 = (0*T,A_xuv2,om_xuv2,s_xuv2,0)

# A_xuv3 = 0.1
# om_xuv3 = 60.57/hbar
# s_xuv3 = 0.03/hbar
# pulse_xuv3 = (0*T,A_xuv3,om_xuv3,s_xuv3,0)

A_probe = 1.0
om_probe = 1.55/hbar
s_probe = 0.20/hbar
pulse_probe = (T,A_probe,om_probe,s_probe,0)

# A_probe2 = 0.1
# om_probe2 = 1.40/hbar
# s_probe2 = 0.04/hbar
# pulse_probe2 = (T,A_probe2,om_probe2,s_probe2,0)

# A_probe3 = 0.1
# om_probe3 = 1.86/hbar
# s_probe3 = 0.03/hbar
# pulse_probe3 = (T,A_probe3,om_probe3,s_probe3,0)

A_ref = 1
om_ref = 1.55/hbar
s_ref = 0.005/hbar
pulse_ref = (0*T,A_ref,om_ref,s_ref,0)

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
            integral_ress[i_t,i_e] = -1j*T_eg*(integral_res + 
                                        hbar*regularization_val*np.log(np.abs((limit_hi-E_val/hbar)/(E_val/hbar-limit_lo))) - 
                                        1j*pi*regularization_val)
    return integral_ress

def spectrum_fun(A,om0,s,om):
    return A/(2*s)*np.exp(-(om-om0)**2/(2*s**2)) # In theory plus negative freq part but it doesnt ever contribute

def sp_tot(gausses,om):
    retval = 0
    for gauss in gausses:
        _,a0,om0,s0,_ = gauss
        retval += spectrum_fun(a0,om0,s0,om)
    return retval

def Amplitude(xuvs,refprobes,E):
    n_t, n_e = E.shape
    amplit_tot = np.zeros((n_t,n_e)).astype(complex)

    for xuv in xuvs:
        for rp in refprobes:
            amplit_tot += Amplitude_ij(xuv,rp,E)
            amplit_tot += Amplitude_ij(rp,xuv,E)
    return amplit_tot

refs = [pulse_ref]
probes = [pulse_probe]#,pulse_probe2,pulse_probe3]
xuvs = [pulse_xuv]#,pulse_xuv2,pulse_xuv3]
refprobes = refs + probes

# om_xuv_range = np.linspace(om_xuv-6*s_xuv,om_xuv+6*s_xuv,N_E)
# plt.plot(om_xuv_range,sp_tot(xuvs, om_xuv_range))
# plt.show()
# om_probe_range = np.linspace(om_probe-6*s_probe,om_probe+6*s_probe,N_E)
# plt.plot(om_probe_range,sp_tot(probes, om_probe_range))
# plt.show()
# om_ref_range = np.linspace(om_ref-6*s_ref,om_ref+6*s_ref,N_E)
# plt.plot(om_ref_range,sp_tot(refs, om_ref_range))
# plt.show()

amplit_tot = Amplitude(xuvs,refprobes,E)

# amplit21_num = Amplitude_ij_num(lambda om: spectrum_fun(A_probe,om_probe,s_probe,om), (om_probe-6*s_probe,om_probe+6*s_probe), 
#                                 lambda om: spectrum_fun(A_xuv,om_xuv,s_xuv,om), (om_xuv-6*s_xuv,om_xuv+6*s_xuv), T_range, 0*T_range, E_range)

plot_mat(amplit_tot,[E_lo,E_hi,-T_reach,T_reach],cmap='plasma',mode='phase')

f_a_21, OM_T, em_lo, em_hi = CFT(T_range,np.abs(amplit_tot)**2)

plot_mat(f_a_21,[E_lo,E_hi,em_lo,em_hi],cmap='plasma',mode='phase')

# correct = correcting_function(OM_T,E,pulse_xuv,pulse_probe,dzeta=0.01)
correct = correcting_function_multi(OM_T,E,pulse_xuv,probes,dzeta=0.001)

corr = correct*f_a_21

rho_lo = 59
rho_hi = 62
fin, small, extent_small = resample(corr,rho_hi,rho_lo,om_ref,E,OM_T,N_T)

# plot_mat(small,extent_small,cmap='plasma',mode='phase')

plot_mat(fin,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')

pop_rho = interpolate.CubicSpline(np.linspace(rho_lo,rho_hi,np.size(np.diag(fin))),np.abs(np.diag(fin)))
pop_zero = interpolate.CubicSpline(E[0,:],np.abs(f_a_21[N_T//2,:])/np.sum(np.abs(f_a_21[N_T//2,:])))

E_range_new = np.linspace(rho_lo,rho_hi,N_E)

plt.plot(E_range_new,pop_rho(E_range_new),label='reconstructed')
plt.plot(E_range_new,pop_zero(E_range_new + hbar*om_ref),label='zero freq')
plt.plot(E_range_new,sp_tot(xuvs, E_range_new/hbar)**2/np.sum(sp_tot(xuvs, E_range_new/hbar)**2),label='original spec')
plt.legend()
plt.show()

# print energy positions of maxima for the plotted peaks
E_rho = E_range_new[np.argmax(pop_rho(E_range_new))]
E_zero = E_range_new[np.argmax(pop_zero(E_range_new + hbar*om_ref))]
spec_vals = sp_tot(xuvs, E_range_new / hbar) ** 2
spec_vals = spec_vals / np.sum(spec_vals)
E_orig = E_range_new[np.argmax(spec_vals)]

print(f"Max of reconstructed (pop_rho): {E_rho:.6f} eV")
print(f"Max of zero-freq reference (pop_zero): {E_zero:.6f} eV")
print(f"Max of original spectrum: {E_orig:.6f} eV")
print(f"Differences: rec-zero = {E_rho - E_zero:.6f} eV, rec-orig = {E_rho - E_orig:.6f} eV")

E1,E2 = np.meshgrid(E_range_new,E_range_new)

mat2 = np.exp(-1/2*((E1/hbar-om_xuv)**2/s_xuv**2 + (E2/hbar-om_xuv)**2/s_xuv**2))
mat2 = mat2/np.sum(np.diag(np.abs(mat2)))
plot_mat(mat2,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')
plot_mat(mat2-fin,[rho_lo,rho_hi,rho_lo,rho_hi],cmap='plasma',mode='phase')