from RK_experiment import RK_experiment, hbar
import numpy as np
import rkraken as rk

def combine_scans(rho_rawA,rho_rawB,rho_raw_sigmaA,rho_raw_sigmaB):
    
    common_area = (np.abs(rho_rawA) > 0) & (np.abs(rho_rawB) > 0)

    WA = 1/(rho_raw_sigmaA**2 + 1e-12)
    WB = 1/(rho_raw_sigmaB**2 + 1e-12)

    rho_rawC = rho_rawA + rho_rawB
    rho_rawC[common_area] = ((WA*rho_rawA + WB*rho_rawB) / (WA + WB))[common_area]

    rho_raw_sigmaC = rho_raw_sigmaA + rho_raw_sigmaB
    rho_raw_sigmaC[common_area] = np.sqrt(1/(WA + WB))[common_area]

    return rho_rawC, rho_raw_sigmaC

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

# Build the density matrix parameters
amps = [1.0/np.sqrt(2), 1.0]
mus = [25.0 - 0.18, 25.0]
sigmas = [0.06, 0.06]
betas = [3, 3]
taus = [1, 1]
lambdas = [5, 5]
gammas = np.array([[1.0, 1.0],
                   [1.0, 1.0]])
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

# Probe pulse definition
A_probe = 0.6
probe_params = {
    'amps': np.asarray([0.3,0.3,0.3]) * A_probe,
    'oms': np.asarray([1.48 / hbar, 1.55 / hbar, 1.62 / hbar]),
    'sigmas': np.asarray([0.04 / hbar, 0.04 / hbar, 0.04 / hbar]),
    'phi0': 0.0,
    'phase_grad': 0.0,
    'phase_chirp': 0.0
}

# Reference pulse definition

A_ref = 1.0
om_ref = 1.52 / hbar
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
    ifWide=False
)

experiment.define_pulses(probe_params)
experiment.define_model(rho_params)
experiment.generate_signal()
experiment.process_and_detrend()
experiment.kb_correct()
# experiment.probe_reconstruct()
rhodata_roiA, rhosigma_roiA, E1A, E2A, rho_rawA, rho_raw_sigmaA = experiment.probe_sp_correct()

# Reference pulse definition

A_ref = 1.0
om_ref = 1.44 / hbar
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
    ifWide=False
)


experiment.define_pulses(probe_params)
experiment.define_model(rho_params)
experiment.generate_signal()
experiment.process_and_detrend()
experiment.kb_correct()
# experiment.probe_reconstruct()
rhodata_roiB, rhosigma_roiB, E1B, E2B, rho_rawB, rho_raw_sigmaB = experiment.probe_sp_correct()
# experiment.mcmc_fit()

notcommon_regionA = ((E2A-E1A) < np.min(E2B-E1B)) | ((E2A-E1A) > np.max(E2B-E1B))
rhodata_roiA_integrand = np.copy(rhodata_roiA)
rhodata_roiA_integrand[notcommon_regionA] = 0

notcommon_regionB = ((E2B-E1B) < np.min(E2A-E1A)) | ((E2B-E1B) > np.max(E2A-E1A))
rhodata_roiB_integrand = np.copy(rhodata_roiB)
rhodata_roiB_integrand[notcommon_regionB] = 0

integralA = np.sum(np.abs(rhodata_roiA_integrand))
integralB = np.sum(np.abs(rhodata_roiB_integrand))

rhodata_roiA = rhodata_roiA * integralA/integralB
rhosigma_roiA = rhosigma_roiA * integralA/integralB

rho_rawC, rho_raw_sigmaC = combine_scans(rho_rawA,rho_rawB,rho_raw_sigmaA,rho_raw_sigmaB)

# rk.plot_mat(rhodata_roiA,saveloc='rhodata_roiA.png')
# rk.plot_mat(rhodata_roiB,saveloc='rhodata_roiB.png')

# rk.plot_mat(rhosigma_roiA,saveloc='rhosigma_roiA.png')
# rk.plot_mat(rhosigma_roiB,saveloc='rhosigma_roiB.png')

# rk.plot_mat(rho_rawC,saveloc='rho_rawC.png')
# rk.plot_mat(rho_raw_sigmaC,saveloc='rho_raw_sigmaC.png')

experiment.rho_raw = rho_rawC
experiment.rho_raw_sigma = rho_raw_sigmaC
experiment.rhodata_roi = np.concatenate(rhodata_roiA.flatten(),rhodata_roiB.flatten())
experiment.rhosigma_roi = np.concatenate(rhosigma_roiA.flatten(),rhosigma_roiB.flatten())
experiment.E1 = np.concatenate(E1A.flatten(),E1B.flatten())
experiment.E2 = np.concatenate(E2A.flatten(),E2B.flatten())

experiment.mcmc_fit()