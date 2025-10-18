import numpy as np
from numpy import pi
from scipy.special import erf
from scipy.special import wofz
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

# Reduced Planck constant in eV*fs (approx CODATA): ħ = 6.582119569e-16 eV·s
hbar = 6.582119569e-1

T_eg = 1 # preliminary simplification that it is const, in atomic units it can be roughly 1-0.01 for 0-100eV

def Amplitude_fun(tau,ene_Eg,A_i,A_j,om0_i,om0_j,s_i,s_j,phi_i,phi_j):
    s = np.sqrt(s_i**2+s_j**2)
    s_t = np.sqrt(s_i**(-2)+s_j**(-2))
    delta = om0_i + om0_j - ene_Eg/hbar
    wofz_arg = s_t/np.sqrt(2)*((om0_j-s_j**2/s**2*delta-1j*tau/s_t**2) - ene_Eg/hbar)

    prefactor = (-pi*A_i*A_j/(4*s_i*s_j)*np.exp(-1j*phi_i*np.sign(om0_i))*np.exp(-1j*phi_j*np.sign(om0_j))*np.exp(1j*om0_i*tau)
                * np.exp(-1/2*( delta**2/s**2 + tau**2/s_t**2 + 2j*s_i/s_j*delta/s*tau/s_t )))
    return prefactor*T_eg*wofz(wofz_arg)

def Feyn_diag_ij(pulse_params_i,pulse_params_j,ene_Eg):
    tau_i,A_i,om0_i,s_i,phi_i = pulse_params_i
    tau_j,A_j,om0_j,s_j,phi_j = pulse_params_j

    return np.exp(1j*ene_Eg/hbar*tau_j)*Amplitude_fun(tau_i-tau_j,ene_Eg,A_i,A_j,om0_i,om0_j,s_i,s_j,phi_i,phi_j)

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

E_lo = 60
E_hi = 64
T_reach = 100

N_E = 600
N_T = 600

E_range = np.linspace(E_lo,E_hi,N_E)
T_range = np.linspace(-T_reach,T_reach,N_T)
E, T = np.meshgrid(E_range,T_range)

A_xuv = 1
om_xuv = 60.65/hbar
s_xuv = 0.15/hbar
pulse_xuv = (0*T,A_xuv,om_xuv,s_xuv,0)

A_probe = 1
om_probe = 1.55/hbar
s_probe = 0.30/hbar
pulse_probe = (T,A_probe,om_probe,s_probe,0)

A_ref = 1
om_ref = 1.55/hbar
s_ref = 0.001/hbar
pulse_ref = (0*T,A_ref,om_ref,s_ref,0)

amplit12 = Feyn_diag_ij(pulse_xuv,pulse_probe,E)
amplit21 = Feyn_diag_ij(pulse_probe,pulse_xuv,E)
amplit13 = Feyn_diag_ij(pulse_xuv,pulse_ref,E)
amplit31 = Feyn_diag_ij(pulse_ref,pulse_xuv,E)

signal = np.abs(amplit31+amplit13+amplit21+amplit12)**2

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

# plot magnitude of the spectrum (rows: energy_axis, cols: E_range)
plt.matshow(np.abs(spec_shift),
                        extent=[E_lo, E_hi, energy_axis_shift[0], energy_axis_shift[-1]],
                        aspect='auto',
                        origin='lower')
plt.xlabel('Final state energy (E)') 
plt.ylabel('Transfer energy (eV)')
plt.title('Windowed CTFT via FFT (per-column over time)')
plt.colorbar(label='|S(ω,E)|')
plt.show()

om_t = energy_axis_shift/hbar
om_t = np.tile(om_t,(N_E,1)).T

correction = correcting_function(om_t,E,pulse_xuv,pulse_probe,dzeta=1e-3,big_sigma=1000)

spec_shift_c = spec_shift*correction

# # plot magnitude of the spectrum (rows: energy_axis, cols: E_range)
# plt.matshow(np.abs(spec_shift_c),
#                         extent=[E_lo, E_hi, energy_axis_shift[0], energy_axis_shift[-1]],
#                         aspect='auto',
#                         origin='lower')
# plt.xlabel('Final state energy (E)') 
# plt.ylabel('Transfer energy (eV)')
# plt.title('Windowed CTFT via FFT (per-column over time)')
# plt.colorbar(label='|S(ω,E)|')
# plt.show()

###
### INTERPOLATION AND AXIS SHIFT
###

# Requested reconstructed matrix range

rho_lo = 60.0
rho_hi = 61.4
rho_valrange = rho_hi-rho_lo

new_Ef_min = rho_lo + hbar*om_ref
new_Ef_max = rho_hi + hbar*om_ref

new_omt_min = om_ref - rho_valrange/hbar
new_omt_max = om_ref + rho_valrange/hbar

idx_min = np.argmin(np.abs(new_Ef_min-E[0,:]))
idx_max = np.argmin(np.abs(new_Ef_max-E[0,:]))

idy_min = np.argmin(np.abs(new_omt_min-om_t[:,0]))
idy_max = np.argmin(np.abs(new_omt_max-om_t[:,0]))

extent = [E[0,idx_min], E[0,idx_max], om_t[idy_min,0], om_t[idy_max,0]]

print([E[0,idx_min], E[0,idx_max], hbar*om_t[idy_min,0], hbar*om_t[idy_max,0]])

small_sig = spec_shift_c[idy_min:idy_max,idx_min:idx_max]

# plot magnitude of the spectrum (rows: energy_axis, cols: E_range)
plt.matshow(np.abs(small_sig),
                        extent=extent,
                        aspect='auto',
                        origin='lower')
plt.xlabel('Final state energy (E)') 
plt.ylabel('Transfer energy (eV)')
plt.title('Windowed CTFT via FFT (per-column over time)')
plt.colorbar(label='|S(ω,E)|')
plt.show()


Ef_range = np.linspace(new_Ef_min,new_Ef_max,idx_max-idx_min)
omt_range = np.linspace(new_omt_min,new_omt_max,idy_max-idy_min)

re_spline = RectBivariateSpline(omt_range*hbar, Ef_range, np.real(small_sig), kx=3, ky=3, s=0)
im_spline = RectBivariateSpline(omt_range*hbar, Ef_range, np.imag(small_sig), kx=3, ky=3, s=0)

N_new = 200
new_Ef_range = np.linspace(new_Ef_min,new_Ef_max,N_new)
new_Et_range = np.linspace(hbar*om_ref - rho_valrange/2,hbar*om_ref + rho_valrange/2,N_new)

ET, EF = np.meshgrid(new_Et_range, new_Ef_range, indexing='ij')

def new_Sig_cc_interp(om_t_vals, Ef_vals, grid=None):
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

EPS1 = EF - ET - new_Ef_min + new_Et_range[-1] + new_omt_min*hbar
EPS2 = EF

Sig_cc_cubic_mesh = new_Sig_cc_interp(EPS1, EPS2)

Sig_cc_cubic_mesh = 1/2*(Sig_cc_cubic_mesh + np.conjugate(Sig_cc_cubic_mesh.T))

# plot magnitude of the spectrum (rows: energy_axis, cols: E_range)
plt.matshow(np.abs(Sig_cc_cubic_mesh),
                        extent=[rho_lo,rho_hi,rho_lo,rho_hi],
                        aspect='auto',
                        origin='lower')
plt.xlabel('Final state energy (E)') 
plt.ylabel('Transfer energy (eV)')
plt.title('Windowed CTFT via FFT (per-column over time)')
plt.colorbar(label='|S(ω,E)|')
plt.show()

###
### NUMERICAL CHECKS
###

# def Amplitude_fun2(tau,ene_Eg,A_i,A_j,om0_i,om0_j,s_i,s_j,phi_i,phi_j):
#     # Parameters
#     s = np.sqrt(s_i**2 + s_j**2)
#     s_t = np.sqrt(s_i**(-2) + s_j**(-2))
#     delta = om0_i + om0_j - ene_Eg/hbar
#     tauE = s_t**2 * (om0_j - ene_Eg/hbar - s_j**2/s**2 * delta)
#     f0 = -pi*A_i*A_j/(4*s_i*s_j)

#     # Stable evaluation of exp(- (tauE/s_t)^2 / 2) * erf( -(tau + i*tauE)/(s_t*sqrt(2)) )
#     # Let z = -(tau + i*tauE)/(s_t*sqrt(2)) = x + i y.
#     # Then: exp(-y^2) * erf(z) = exp(-y^2) - exp(-x^2) * exp(-i*2*x*y) * wofz(i*z)
#     rt2 = np.sqrt(2.0)
#     z = -(tau + 1j*tauE) / (s_t * rt2)
#     x = np.real(z)
#     y = np.imag(z)

#     term1 = np.exp(-y*y)                       # exp(- (tauE/s_t)^2 / 2)
#     term2 = np.exp(-x*x) * np.exp(-1j*2*x*y) * wofz(1j*z)
#     scaled_erf = term1 - term2                 # equals exp(-y^2) * erf(z)

#     # Multiply remaining smooth factor exp(- (delta/s)^2 / 2)
#     return -T_eg * f0 * scaled_erf * np.exp(-0.5 * (delta/s)**2)

# def spect0(om_t,ene_Eg,A_i,A_j,om0_i,om0_j,s_i,s_j,phi_i,phi_j):
#     # Parameters
#     s = np.sqrt(s_i**2 + s_j**2)
#     s_t = np.sqrt(s_i**(-2) + s_j**(-2))
#     delta = om0_i + om0_j - ene_Eg/hbar
#     tauE = s_t**2 * (om0_j - ene_Eg/hbar - s_j**2/s**2 * delta)
#     f0 = -pi*A_i*A_j/(4*s_i*s_j)

#     return -T_eg * f0 * 2j / (om_t+0.0001) * np.exp(-1/2 * ((s_t*om_t + tauE/s_t)**2 + (delta/s)**2))


# def spect(pulse_params_i,pulse_params_j,ene_Eg,om_t):
#     tau_i,A_i,om0_i,s_i,phi_i = pulse_params_i
#     tau_j,A_j,om0_j,s_j,phi_j = pulse_params_j

#     return spect0(om_t,ene_Eg,A_i,A_j,om0_i,om0_j,s_i,s_j,phi_i,phi_j)


# om_t = energy_axis_shift/hbar
# om_t = np.tile(om_t,(N_E,1)).T
# print(om_t.shape)

# # spec_an = spect(pulse_probe,pulse_xuv,E,om_t)

# # # plot magnitude of the spectrum (rows: energy_axis, cols: E_range)
# # plt.matshow(np.abs(spec_an),
# #             extent=[E_lo, E_hi, energy_axis_shift[0], energy_axis_shift[-1]],
# #             aspect='auto',
# #             origin='lower')
# # plt.xlabel('Final state energy (E)') 
# # plt.ylabel('Transfer energy (eV)')
# # plt.title('Windowed FFT magnitude (per-column FFT over time)')
# # plt.colorbar(label='|S(ω,E)|')
# # plt.show()

# def spectt0(om_t,ene_Eg,A_x,A_p,A_r,om0_x,om0_p,om0_r,s_x,s_p,s_r):
#     # Parameters
#     s = np.sqrt(s_x**2 + s_p**2)
#     s_t = np.sqrt(s_x**(-2) + s_p**(-2))
#     delta_p = om0_x + om0_p - ene_Eg/hbar
#     delta_r = om0_x + om0_r - ene_Eg/hbar
#     tauE = s_t**2 * (om0_x - ene_Eg/hbar - s_x**2/s**2 * delta_p)
#     f0_p = -pi*A_x*A_p/(4*s_x*s_p)
#     f0_r = -pi*A_x*A_r/(4*s_x*s_r)

#     return T_eg**2 * f0_p * f0_r * s_r * 4 * np.sqrt(2/pi) / (om_t+0.00001) / om0_r * np.exp(-1/2 * ((s_t*om_t + tauE/s_t)**2 + (delta_r/s_x)**2))

# def spectt(pulse_params_x,pulse_params_p,pulse_params_r,ene_Eg,om_t):
#     tau_x,A_x,om0_x,s_x,phi_x = pulse_params_x
#     tau_p,A_p,om0_p,s_p,phi_p = pulse_params_p
#     tau_r,A_r,om0_r,s_r,phi_r = pulse_params_r

#     return spectt0(om_t,ene_Eg,A_x,A_p,A_r,om0_x,om0_p,om0_r,s_x,s_p,s_r)

# spectt_an = spectt(pulse_xuv,pulse_probe,pulse_ref,E,om_t) + spectt(pulse_xuv,pulse_probe,pulse_ref,E,-1*om_t)

# # plot magnitude of the spectrum (rows: energy_axis, cols: E_range)
# plt.matshow(np.abs(spectt_an),
#             extent=[E_lo, E_hi, energy_axis_shift[0], energy_axis_shift[-1]],
#             aspect='auto',
#             origin='lower')
# plt.xlabel('Final state energy (E)') 
# plt.ylabel('Transfer energy (eV)')
# plt.title('Windowed FFT magnitude (per-column FFT over time)')
# plt.colorbar(label='|S(ω,E)|')
# plt.show()

# # plot magnitude of the spectrum (rows: energy_axis, cols: E_range)
# plt.matshow(np.log(np.maximum(np.abs(spectt_an - spec_shift),0.001)),
#             extent=[E_lo, E_hi, energy_axis_shift[0], energy_axis_shift[-1]],
#             aspect='auto',
#             origin='lower')
# plt.xlabel('Final state energy (E)') 
# plt.ylabel('Transfer energy (eV)')
# plt.title('Windowed FFT magnitude (per-column FFT over time)')
# plt.colorbar(label='|S(ω,E)|')
# plt.show()