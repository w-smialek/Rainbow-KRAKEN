import numpy as np
from numpy import pi
from scipy.special import wofz
import matplotlib.pyplot as plt

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
s_xuv = 0.25/hbar
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
# plt.matshow(np.abs(amplit12))
# plt.show()

amplit21 = Feyn_diag_ij(pulse_probe,pulse_xuv,E)
# plt.matshow(np.abs(amplit21))
# plt.show()

amplit13 = Feyn_diag_ij(pulse_xuv,pulse_ref,E)
plt.matshow(np.imag(amplit13))
plt.show()

amplit31 = Feyn_diag_ij(pulse_ref,pulse_xuv,E)
plt.matshow(np.imag(amplit31))
plt.show()

print(np.sum(np.abs(amplit12)))
print(np.sum(np.abs(amplit21)))
print(np.sum(np.abs(amplit13)))
print(np.sum(np.abs(amplit31)))

signal = np.abs(amplit31+amplit13+amplit21+amplit12)**2

plt.matshow(signal)
plt.show()

# apply Hanning window along time (rows) for each column (energy)
dt = T_range[1] - T_range[0]
window = np.hanning(N_T)[:, None]  # column vector to multiply each column
windowed = signal * window

# Fourier transform along time axis (rows). multiply by dt to approximate continuous FT.
spec = np.fft.fft(windowed, axis=0) * dt
spec_shift = np.fft.fftshift(spec, axes=0)

# build frequency / energy axis (angular frequency -> energy using hbar)
freqs = np.fft.fftfreq(N_T, d=dt)        # cycles per unit time
omega = 2 * np.pi * freqs                # rad / unit time
energy_axis = hbar * omega               # energy axis in same units as hbar
energy_axis_shift = np.fft.fftshift(energy_axis)

# plot magnitude of the spectrum (rows: energy_axis, cols: E_range)
plt.matshow(np.abs(spec_shift)**0.5,
            extent=[E_lo, E_hi, energy_axis_shift[0], energy_axis_shift[-1]],
            aspect='auto',
            origin='lower')
plt.xlabel('Final state energy (E)') 
plt.ylabel('Transfer energy (eV)')
plt.title('Windowed FFT magnitude (per-column FFT over time)')
plt.colorbar(label='|S(ω,E)|')
plt.show()