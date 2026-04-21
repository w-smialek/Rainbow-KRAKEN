from RK_experiment import RK_experiment, hbar


E_lo = 24.5
E_hi = 28.0
T_reach = 150
E_res = 0.01
N_T = 501
p_E = 4
alpha = 10000
b = 1

sideband_lo = 25.5
sideband_hi = 28.0
harmq_lo = 24.5
harmq_hi = 25.5

if_coherent = True

# Probe pulse definition (latest configuration from WF_pipeline_zero.py).
A_probe = 0.6
a_probes = [1.0, 0.3]
om_probes = [1.55 / hbar, 1.66 / hbar]
s_probes = [0.10 / hbar, 0.05 / hbar]
probe_phase = 1.0
probe_phase_grad = 2.5
probe_phase_chirp = 2.5

A_ref = 1.0
om_ref = 1.45 / hbar
s_ref = 0.025 / hbar

experiment = RK_experiment(
    E_lo=E_lo,
    E_hi=E_hi,
    T_reach=T_reach,
    E_res=E_res,
    N_T=N_T,
    p_E=p_E,
    alpha=alpha,
    b=b,
    sb_lo=sideband_lo,
    sb_hi=sideband_hi,
    harmq_lo=harmq_lo,
    harmq_hi=harmq_hi,
    if_coherent=if_coherent,
    A_ref=A_ref,
    om_ref=om_ref,
    s_ref=s_ref,
)

experiment.ifWF = False
experiment.ifWide = False

experiment.define_pulses(
    A_probe,
    a_probes,
    om_probes,
    s_probes,
    probe_phase=probe_phase,
    probe_phase_grad=probe_phase_grad,
    probe_phase_chirp=probe_phase_chirp,
)

experiment.define_model()
experiment.generate_signal()
experiment.process_and_detrend()
experiment.kb_correct()
experiment.probe_reconstruct()
experiment.probe_sp_correct()
# experiment.mcmc_fit()