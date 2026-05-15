from RK_experiment_EXP import RK_experiment, hbar
import numpy as np

E_counts = np.atleast_2d(np.loadtxt('scan_data/Ecounts.csv', delimiter=',', dtype=float))
E_bins = np.ravel(np.loadtxt('scan_data/E_bins.csv', delimiter=',', dtype=float))

E_lo_cut = 10.5
E_hi_cut = 14

energy_mask = (E_bins >= E_lo_cut) & (E_bins <= E_hi_cut)
E_bins = E_bins[energy_mask]
E_counts = E_counts[:, energy_mask]

N_zero_pad = 0# 2*250
E_counts = np.pad(E_counts, ((N_zero_pad // 2, N_zero_pad // 2), (0, 0)), mode='constant')

E_lo = E_bins[0]
E_hi = E_bins[-1]
N_T = 251
T_reach = 50
T_reach = T_reach*(N_T + N_zero_pad)/(N_T)
N_T = N_T + N_zero_pad
E_res = 0.025
alpha = 10000
b = 0

sideband_lo = 12.5
sideband_hi = 14.0
harmq_lo = 10.5
harmq_hi = 12.5

A_ref = 0.5
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
)

experiment.median_filter_when = 1
experiment.prrec_lambda1 = 0.2
experiment.prrec_lambda2 = 0.01
experiment.mcmc_num_warmup = 3000


# Probe pulse definition
A_probe = 0.6
probe_params = {
    'amps': np.asarray([1.0]) * A_probe,
    'oms': np.asarray([1.55 / hbar]),
    'sigmas': np.asarray([0.035 / hbar]),
    'phi0': 0.0,
    'phase_grad': 0.0,
    'phase_chirp': 0.0
}

# Build the density matrix parameters
amps = [1.0/np.sqrt(2), 1.0]
mus = [11.62 - 0.18, 11.62]
sigmas = [0.10, 0.10]
betas = [0, 0]
taus = [0, 0]
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
experiment.generate_signal(exp_signal=E_counts)
experiment.process_and_detrend()
experiment.kb_correct()
# experiment.probe_reconstruct()
# experiment.probe_sp_correct()
# experiment.mcmc_fit()