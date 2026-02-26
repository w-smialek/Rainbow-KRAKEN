import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import rkraken as rk
from WF_pipeline import RK_experiment
from scipy.ndimage import gaussian_filter1d

# Reduced Planck constant in eV*fs (approx CODATA): ħ = 6.582119569e-16 eV·s
hbar = 6.582119569e-1

# Read CSV files into numpy arrays
delay = np.loadtxt('scan_data/delay.csv', delimiter=',')
E_bins = np.loadtxt('scan_data/E_bins.csv', delimiter=',')
Ecounts = np.loadtxt('scan_data/Ecounts.csv', delimiter=',')

b = np.mean(Ecounts[:,0:10])
print(b)
S = (Ecounts - b) / np.sum(Ecounts, axis=1, keepdims=True)

S = np.astype(S*500,int)

S = Ecounts - b

row_sums = np.sum(S, axis=1)
sigma = len(row_sums) / 50
row_sums = gaussian_filter1d(row_sums, sigma=sigma)
avg_row_sum = np.mean(row_sums)
S = np.astype(S / row_sums[:, np.newaxis] * avg_row_sum, int)

E_bins = E_bins[200:600]
S = S[:,200:600]

print(f"delay shape: {delay.shape}")
print(f"E_bins shape: {E_bins.shape}")
print(f"Ecounts shape: {Ecounts.shape}")

rk.plot_mat(Ecounts,extent=[E_bins[0],E_bins[-1],delay[0],delay[-1]],saveloc='sc.png')


# plt.plot(np.sum(Ecounts,axis=1))
# plt.savefig('scc.png')


# Example usage of the RK_experiment class
if __name__ == "__main__":

    E_lo = E_bins[0]
    E_hi = E_bins[-1]
    T_reach = 50
    E_res = 0.025    
    N_T = 251
    p_E = 4  # N_E upsampling integer
    alpha = 10000
    # b = 1

    sideband_lo = 9.0
    sideband_hi = 11.0
    harmq_lo = 7.5

    # sideband_lo = 5.9
    # sideband_hi = 7.4
    # harmq_lo = 4.4

    if_coherent = False

    # Create experiment instance
    experiment = RK_experiment(E_lo=E_lo,E_hi=E_hi,T_reach=T_reach,E_res=E_res,N_T=N_T,p_E=p_E,alpha=alpha,b=b,
                            sb_lo=sideband_lo,sb_hi=sideband_hi,harmq_lo=harmq_lo,if_coherent=if_coherent,E_range=E_bins)
    
    experiment.zero_pad = 200
    
    # Define pulses

    A_xuv = 0.1
    a_xuvs = [np.sqrt(0.5),np.sqrt(1.0)]
    om_xuvs = [(10.0-1*1.63)/hbar,(10.0-1*1.63 + 0.17)/hbar]
    s_xuvs = [0.17/hbar,0.17/hbar]

    A_probe = 1.2
    a_probes = [1.0]
    om_probes = [1.55/hbar]
    s_probes = [0.15/hbar]

    A_ref = 0.6
    a_refs = [1.0]
    om_refs = [1.55/hbar]
    s_refs = [0.05/hbar]

    experiment.define_pulses(A_xuv,A_probe,A_ref,a_xuvs,a_probes,a_refs,om_xuvs,om_probes,om_refs,s_xuvs,s_probes,s_refs)
    
    # Run the full pipeline
    experiment.generate_signal(exp_signal=S)

    experiment.process_and_detrend()
    experiment.kb_correct()
    # experiment.xuv_peak()
    experiment.WF_reconstruct()
    experiment.resample_analyze()