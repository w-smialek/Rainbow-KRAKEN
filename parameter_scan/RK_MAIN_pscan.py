from pathlib import Path
import sys

# Allow imports from the repository root when this script is run directly.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from RK_experiment import RK_experiment, hbar
import numpy as np

# E_lo = 24.5
# E_hi = 28.0
# T_reach = 150
# E_res = 0.01
# N_T = 501
# alpha = 5000
# b = 10

# arr = np.load('./parameter_scan/scan_full.npy')

fids = []

k = 2.5
ttot = 0.4
# ttots = np.linspace(0.3,1.0,8)
# kslog2 = np.linspace(-2.5,2.5,20)
# ks = 2**(kslog2)
sigma_rs = np.linspace(0.010,0.080,9)/hbar

# sigma_rs = np.linspace(0.063,0.080,5)/hbar

for sigma_r in sigma_rs:
        suffix = f'_sr_{sigma_r*hbar:.4f}'

        print(f'===== {suffix} =====')
        E_lo = 24.5
        E_hi = 28.0
        T_reach = 100 * k
        E_res = 0.01
        N_T = int(201 * k)//2 * 2 + 1
        alpha = 10000 / k * ttot
        b = 10

        sideband_lo = 25.5
        sideband_hi = 28.0
        harmq_lo = 24.5
        harmq_hi = 25.5

        # Probe pulse definition
        A_probe = 0.6
        probe_params = {
            'amps': np.asarray([1.0]) * A_probe,
            'oms': np.asarray([1.55 / hbar]),
            'sigmas': np.asarray([0.08 / hbar]),
            'phi0': 0.0,
            'phase_grad': 0.0,
            'phase_chirp': 0.0
        }

        # Build the density matrix parameters
        amps = [1.0/np.sqrt(2), 1.0]
        mus = [25.0 - 0.18, 25.0]
        sigmas = [0.06, 0.06]
        betas = [3, 3]
        taus = [1, 1]
        lambdas = [0, 0]
        gammas = np.array([[1.0, 0.0],
                        [0.0, 1.0]])
        etas = np.array([[1.0, 0.0],
                        [0.0, 1.0]])
        # Build the density matrix parameters
        amps = [1.0, 1.0]
        mus = [25.0 - 0.12, 25.0]
        sigmas = [0.06, 0.06]
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

        A_ref = 1.0
        om_ref = 1.50 / hbar
        # s_ref = 0.025 / hbar
        s_ref = sigma_r

        # # s_ref_list = np.linspace(0.010,0.050,9) / hbar
        # s_ref_list = np.linspace(0.060,0.080,3) / hbar
        # A_ref_list = np.ones_like(s_ref_list) * 1.0
        # om_ref_list = np.ones_like(s_ref_list) * 1.50 / hbar

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
        experiment.mcmc_peaks = 2
        experiment.mcmc_num_chains = 1

        experiment.define_pulses(probe_params)
        experiment.define_model(rho_params)
        experiment.generate_signal()#(suffix=suffix)
        experiment.process_and_detrend()
        experiment.kb_correct()
        # experiment.probe_reconstruct()
        experiment.probe_sp_correct(suffix=suffix)
        fid, _ = experiment.mcmc_fit(suffix=suffix)
        fids.append(fid)

        print(fids)
        np.save(f'parameter_scan/scan_sr.npy',np.array(fids))