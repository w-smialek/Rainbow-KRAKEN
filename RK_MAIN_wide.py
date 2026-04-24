from RK_experiment import RK_experiment, hbar
import numpy as np

E_lo = 24.5
E_hi = 28.0
T_reach = 150
E_res = 0.01
N_T = 501
alpha = 30000
b = 1

sideband_lo = 25.5
sideband_hi = 28.0
harmq_lo = 24.5
harmq_hi = 25.5

A_ref = 1.0
om_ref = 1.55 / hbar
s_ref = 0.025 / hbar

experiment = RK_experiment(
    E_lo=E_lo,
    E_hi=E_hi,
    T_reach=T_reach,
    E_res=E_res,
    N_T=N_T,
    alpha=alpha,
    b=b,
    sb_lo=sideband_lo,
    sb_hi=sideband_hi,
    harmq_lo=harmq_lo,
    harmq_hi=harmq_hi,
    A_ref=A_ref,
    om_ref=om_ref,
    s_ref=s_ref,
    ifWide=True
)

# Probe pulse definition
A_probe = 0.6
probe_params = {
    'amps': np.asarray([1.0]) * A_probe,
    'oms': np.asarray([1.55 / hbar]),
    'sigmas': np.asarray([0.18 / hbar]),
    'phi0': 0.0,
    'phase_grad': 0.0,
    'phase_chirp': 0.0
}

# Build the density matrix parameters
amps = [1.0/np.sqrt(2), 1.0]
mus = [25.0 - 0.18 + om_ref * hbar, 25.0 + om_ref * hbar]
sigmas = [0.08, 0.08]
betas = [3, 3]
taus = [1, 1]
lambdas = [0, 0]
gammas = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
etas = np.array([[1.0, 0.0],
                 [0.0, 1.0]])

rho_params = {
    'amps': np.asarray(amps, dtype=float),
    'mus': np.asarray(mus, dtype=float),
    'sigmas': np.asarray(sigmas, dtype=float),
    'betas': np.asarray(betas, dtype=float),
    'taus': np.asarray(taus, dtype=float),
    'lambdas': np.asarray(lambdas, dtype=float),
    'gammas': np.asarray(gammas, dtype=np.complex128),
    'etas': np.asarray(etas, dtype=float),
}

experiment.define_pulses(probe_params)
experiment.define_model(rho_params)
experiment.generate_signal()
experiment.process_and_detrend()
experiment.kb_correct()
experiment.probe_reconstruct()
experiment.probe_sp_correct()
experiment.mcmc_fit()