"""
Rainbow-KRAKEN WF Pipeline - Refactored Class-based Implementation

This module contains the RK_experiment class which encapsulates the full 
Rainbow-KRAKEN signal processing pipeline for quantum state reconstruction.

The pipeline consists of the following stages:
1. prepare_grid() - Initialize energy/time grids and pulse configurations
2. generate_signal() - Generate synthetic baseline and full signal with noise
3. process_and_detrend() - Process signal, estimate background, and detrend
4. kb_correct() - Apply Koay-Basser correction for Rician noise
5. xuv_peak() - Reconstruct XUV peak spectrum 
6. WF_reconstruct() - Retrieve probe spectrum using Wirtinger Flow
7. resample_analyze() - Resample and analyze to produce final density matrices

Usage:
    experiment = RK_experiment()
    experiment.generate_signal()
    experiment.process_and_detrend() 
    experiment.kb_correct()
    experiment.xuv_peak()
    experiment.WF_reconstruct()
    experiment.resample_analyze()

All experimental parameters are stored as class attributes and can be 
modified before running the pipeline methods.
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import median_filter
import rkraken1 as rk

# Reduced Planck constant in eV*fs (approx CODATA): ħ = 6.582119569e-16 eV·s
hbar = 6.582119569e-1

def rho_peak(x,mu,sigma,beta,tau):
    return 1/(2*pi*sigma**2)**(0.25) * np.exp(-(1/(4*sigma**2) - 1j*beta/2)*(x - mu)**2 + 1j*tau*(x - mu))

def rho_model(e1,e2,amps,mus,sigmas,betas,taus,lambdas,gammas,etas):
    retval = 0
    for k in range(len(mus)):
        for l in range(len(mus)):
            D_kl = np.exp( - lambdas[k]**2/2 * (e1 - mus[k])**2 - lambdas[l]**2/2 * (e2 - mus[l])**2 + etas[k,l] * lambdas[k] * lambdas[l] * (e1 - mus[k]) * (e2 - mus[l]))
            retval += (amps[k] * amps[l] * gammas[k,l] * rho_peak(e1,mus[k],sigmas[k],betas[k],taus[k]) 
                       * np.conj(rho_peak(e2,mus[l],sigmas[l],betas[l],taus[l])) * D_kl)

    return retval

def plot_spectra(om_pr,om_x,sp_pr,sp_ref,sp_x):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot: Probe and Reference spectra
    ax1.plot(om_pr * hbar, sp_pr, label='Probe spectrum', linewidth=2)
    ax1.plot(om_pr * hbar, sp_ref, label='Reference spectrum', linewidth=2)
    ax1.set_xlabel('Energy [eV]',fontweight='bold')
    ax1.set_ylabel('Amplitude [arb. u.]',fontweight='bold')
    ax1.set_title('Probe and Reference Spectra',fontweight='bold',fontsize=12)
    ax1.set_xlim([1,2])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right subplot: XUV spectrum
    ax2.plot(om_x * hbar, sp_x, label='Photoelectron populations', linewidth=2, color='purple')
    ax2.set_xlabel('Energy [eV]',fontweight='bold')
    ax2.set_ylabel('Signal strength [arb. u.]',fontweight='bold')
    ax2.set_title('Photoelectron populations',fontweight='bold',fontsize=12)
    ax2.set_xlim([24,26])
    ax2.grid(True, alpha=0.3)
    fig.suptitle('IR spectrum and photelectron signal', fontsize=20, weight='bold')
    
    plt.tight_layout()
    plt.savefig('single_output_temp/spectra/input_spectra.png', dpi=300)
    plt.close()

    return

class RK_experiment:
    """Rainbow-KRAKEN experiment class for full pipeline processing."""
    
    def __init__(self,E_lo=60.0,E_hi=63.5,T_reach=50,E_res=0.025,N_T=240,p_E=4,alpha=0.05,b=1,sb_lo=24.7,sb_hi=28,harmq_lo=22,harmq_hi=24.7,
                 if_coherent=True,mask_lo=0.8,mask_hi=2.2,donor_lo=2.5,E_range=None,A_ref=1,om_ref=1.55/hbar,s_ref=0.02/hbar):

        self.ifexp = False
        self.ifWF = True
        self.ifWide = False

        # Field parameters
        self.E_lo = E_lo
        self.E_hi = E_hi
        self.T_reach = T_reach
        self.E_span = self.E_hi - self.E_lo
        
        self.E_res = E_res
        self.N_E = round(self.E_span/self.E_res/10)*10
        
        self.N_T = N_T
        self.p_E = p_E  # N_E upsampling integer
        
        self.alpha = alpha
        self.b = b
        
        # Coherence flag
        self.if_coherent = if_coherent
        
        # Noise and processing parameters
        self.ifnoise = True
        self.sb_lo = sb_lo
        self.sb_hi = sb_hi
        self.harmq_lo = harmq_lo
        self.harmq_hi = harmq_hi
        self.zero_pad = 0
        self.window_zerocomp = False
        
        # Detrending parameters
        self.em_axis_mid_reach = 1.00
        self.T_mix_reach = 30
        self.mask_lo = mask_lo   # indirect-energy lower edge of the feature band (eV)
        self.mask_hi = mask_hi   # indirect-energy upper edge of the feature band (eV)
        self.donor_lo = donor_lo  # lower edge of the feature-free donor band (eV)
        
        # Correction parameters
        self.WF_eps = 2e-9
        self.WF_eps = 1e-8
        self.dzeta_val = 1e-3
        self.theta_val = 0.01
        
        # Band parameters
        self.rho_lo = harmq_lo
        self.rho_hi = sb_lo

        # filter parameters
        self.A_ref = A_ref
        self.om_ref = om_ref
        self.s_ref = s_ref
        
        # Random number generator
        self.rng = np.random.default_rng()
        
        # Initialize grids and pulses
        self.prepare_grid(E_range=None)
    
    def prepare_grid(self,E_range=None):
        """Initialize energy and time grids and pulse configurations."""
        if E_range is None:
            self.E_range = np.linspace(self.E_lo, self.E_hi, self.N_E)
        else:
            self.E_range = E_range
        self.T_range = np.linspace(-self.T_reach, self.T_reach, self.N_T)
        self.E, self.T = np.meshgrid(self.E_range, self.T_range)
        
        self.N_E_up = self.p_E * self.N_E
        self.E_up_range = np.linspace(self.E_lo, self.E_hi, self.N_E_up)
        self.E_up, self.T_up = np.meshgrid(self.E_up_range, self.T_range)

    def define_pulses(self,A_probe,a_probes,om_probes,s_probes):
        # Define probe pulse configurations (time-dependent, tau=T)
        probes_list = []
        for i in range(len(a_probes)):
            probe_pulse = (self.T, A_probe*a_probes[i], om_probes[i], s_probes[i], 0)
            probes_list.append(probe_pulse)
        self.probes = tuple(probes_list)

    def define_model(self):
        
        amps = [1.0/np.sqrt(2),1.0]
        mus = [25.0-0.18+self.om_ref*hbar,25.0+self.om_ref*hbar]
        sigmas = [0.08,0.08]
        betas = [3,3]
        taus = [1,1]
        lambdas = [0,0]
        gammas = np.array([[1.0,0.0],
                           [0.0,1.0]])
        etas = np.array([[1.0,0.0],
                         [0.0,1.0]])

        # Keep known rho parameters for probe-only Bayesian inference.
        self.rho_params = {
            'amps': np.asarray(amps, dtype=float),
            'mus': np.asarray(mus, dtype=float),
            'sigmas': np.asarray(sigmas, dtype=float),
            'betas': np.asarray(betas, dtype=float),
            'taus': np.asarray(taus, dtype=float),
            'lambdas': np.asarray(lambdas, dtype=float),
            'gammas': np.asarray(gammas, dtype=np.complex128),
            'etas': np.asarray(etas, dtype=float),
        }

        def rho_f(e1, e2):
            return rho_model(e1,e2,amps,mus,sigmas,betas,taus,lambdas,gammas,etas) 

        self.rho_f = rho_f

    def generate_signal(self,exp_signal=None):
        """Generate synthetic baseline and full signal with optional noise."""

        ### GENERATE SPECTRA

        ir_lo,xuv_lo,ir_hi,xuv_hi = rk.conv_bounds(self.E_lo/hbar,self.E_hi/hbar,self.N_E,self.om_ref)
        self.om_probe = np.linspace(ir_lo,ir_hi,self.N_E)
        self.om_xuv = np.linspace(xuv_lo,xuv_hi,self.N_E)

        ref_mask = self.A_ref*((self.om_probe >= self.om_ref - self.s_ref) & (self.om_probe <= self.om_ref + self.s_ref)).astype(float)

        self.sp_probe = rk.sp_tot(self.probes, self.om_probe)
        self.sp_ref = self.sp_probe*ref_mask

        ### PLOT the upsampled spectra with probe/ref on one axis, xuv on another

        plot_spectra(self.om_probe,self.om_xuv,self.sp_probe,self.sp_ref,np.abs(self.rho_f(self.om_xuv*hbar + self.om_ref*hbar,self.om_xuv*hbar + self.om_ref*hbar)))
        
        ### SIMULATE SIGNAL

        _, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, np.zeros((self.N_T,self.N_E)), use_window=False, zero_pad=self.zero_pad)

        OM_P = np.tile(self.om_probe, (self.N_T, 1))
        OM_X = np.tile(self.om_xuv, (self.N_T, 1))

        a_ref = self.A_ref * rk.sp_tot(self.probes,self.om_ref) / self.om_ref
        a_pr = rk.sp_tot(self.probes,self.OM_T) / rk.regularize_omega(self.OM_T)

        d_en = (self.om_probe[1] - self.om_probe[0])*hbar
        omega_cols = (np.arange(self.N_E) - (self.N_E - 1) / 2.0) * d_en
        psf_row_mask = ((omega_cols < self.s_ref*hbar) & (omega_cols > -self.s_ref*hbar)).astype(float)
        PSF_ref = np.tile(psf_row_mask, (self.N_T, 1))
        PSF_ref2 = np.tile(psf_row_mask, (self.N_E, 1))

        i_om0 = np.argmin(np.abs(self.OM_T[:,0]))

        E1,E2 = np.meshgrid(self.E[0,:],self.E[0,:])

        rho_shifted = self.rho_f(self.E-self.OM_T*hbar+self.om_ref*hbar, self.E)
        signal_ft0 = np.conj(a_ref)*a_pr*fftconvolve(rho_shifted,PSF_ref,mode='same',axes=1)*d_en
        signal_ft0 += np.conj(np.flip(signal_ft0,axis=0))

        rho_ss = fftconvolve(
            fftconvolve(self.rho_f(E1,E2),PSF_ref2,mode='same',axes=1),
            PSF_ref2.T,
            mode='same',
            axes=0
        )*d_en**2
        rho_ss_diag = np.diag(rho_ss)

        signal_ft0[i_om0, :] += np.abs(a_ref)**2 * rho_ss_diag * self.T_reach*2

        in_1 = self.rho_f((OM_X - self.OM_T + self.om_ref)*hbar, (OM_X + self.om_ref)*hbar)
        in_2_denom = (OM_P + self.OM_T) * OM_P # CAN PRODUCE DIVISION BY ZERO!!
        in_2_num = (rk.sp_tot(self.probes, OM_P + self.OM_T)
            * np.conj(rk.sp_tot(self.probes,OM_P)))
        in_2 = (in_2_num
                / in_2_denom)
        
        d_om = OM_P[0,1] - OM_P[0,0]
        zerocomp = fftconvolve(in_1,in_2,axes=1,mode='same') * d_om

        signal_ft0 += zerocomp

        self.signal_ft0 = self.alpha*signal_ft0


        signal_clean, _, _, _ = rk.CFT(self.T_range, signal_ft0, use_window=False, zero_pad=0, inverse=True)
        signal_clean = np.abs(signal_clean)

        rk.plot_mat(self.signal_ft0-1e-3,extent=[self.E_lo,self.E_hi,em_lo,em_hi],saveloc='single_output_temp/pipeline_diag/model_ft.png')
        rk.plot_mat(signal_clean-1e-6,extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],saveloc='single_output_temp/pipeline_diag/model_sig.png')

        ### APPLY NOISE PROCEDURE

        signal_clean *= self.alpha
        signal_clean += self.b
        if not self.ifnoise:
            self.signal = signal_clean.copy()
        else:
            # Apply Poisson noise assuming signal_clean contains expected values (means)
            self.signal = self.rng.poisson(signal_clean).astype(float)
        self.signal -= self.b # NOISE FLOOR SHOULD BE QUITE PRECISELY MEASURED BY CAPTURING WITH LASER OFF

        ### CUT INTO MB AND SB

        sb_lo_idx = np.argmin(np.abs(self.E_range - self.sb_lo))
        sb_hi_idx = np.argmin(np.abs(self.E_range - self.sb_hi))

        self.signal_sb = np.zeros_like(self.signal)
        self.signal_sb[:, sb_lo_idx:sb_hi_idx+1] = self.signal[:, sb_lo_idx:sb_hi_idx+1]

        harmq_lo_idx = np.argmin(np.abs(self.E_range - self.harmq_lo))
        harmq_hi_idx = np.argmin(np.abs(self.E_range - self.harmq_hi))

        self.signal_harmq = np.zeros_like(self.signal)
        self.signal_harmq[:, harmq_lo_idx:harmq_hi_idx+1] = self.signal[:, harmq_lo_idx:harmq_hi_idx+1]

        ### PLOT FULL SIGNAL AND BANDS

        peak_sb_counts = int(np.max(self.signal_sb))
        self.peak_sb_counts = peak_sb_counts

        rk.plot_mat(self.signal, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
                caption='Peak SB counts: %i\nN_T: %i\nN_E: %i\nT_res: %.2f fs\nE_res: %.3f eV'%(
                    self.peak_sb_counts,self.N_T,self.N_E,2*self.T_reach/self.N_T,self.E_res),
                saveloc='single_output_temp/pipeline_diag/measured_signal.png',xlabel='Kinetic energy $E_f$ (eV)',ylabel='Time delay $\\tau$ (fs)',mode='abs',title='Simulated noisy signal')

    def process_and_detrend(self):
        
        self.signal_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal, use_window=False, zero_pad=self.zero_pad)
        self.signal_sb_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal_sb, use_window=False, zero_pad=self.zero_pad)

        rk.plot_mat(self.signal_FT, extent=[self.E_lo,self.E_hi,em_lo,em_hi],
                saveloc='single_output_temp/pipeline_diag/signal_FT.png', show=False)
        
        ### ROI SLICE FRACTS

        self.x_lo = 24.0 + self.om_ref*hbar
        self.x_hi = 26.0 + self.om_ref*hbar

        self.y_lo = 1.0/hbar
        self.y_hi = 2.0/hbar

        Ef_reach = self.E[0,-1] - self.E[0,1]
        self.Ef_slice_fracts = ((self.x_lo - self.E[0,0])/Ef_reach, (self.x_hi - self.E[0,0])/Ef_reach)

        OMt_reach = self.OM_T[-1,0] - self.OM_T[1,0]
        self.OMt_slice_fracts = ((self.y_lo - self.OM_T[1,0])/OMt_reach, (self.y_hi - self.OM_T[1,0])/OMt_reach)

        self.signal_ft0_ROI, em_axis_mid, i0, i1, ef_axis_mid, e0, e1 = rk.extract_midslice(self.signal_ft0, self.OMt_slice_fracts, hbar*self.OM_T[:,0], self.Ef_slice_fracts, self.E)
        signal_sb_FT_ROI, em_axis_mid, i0, i1, ef_axis_mid, e0, e1 = rk.extract_midslice(self.signal_sb_FT, self.OMt_slice_fracts, hbar*self.OM_T[:,0], self.Ef_slice_fracts, self.E)
        
        rk.plot_mat(signal_sb_FT_ROI, extent=[self.x_lo,self.x_hi,self.y_lo*hbar,self.y_hi*hbar],
                saveloc='single_output_temp/pipeline_diag/signal_FT_sb_ROI.png', xlabel='Kinetic energy $E_f$ (eV)',ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)', show=False,title='Sideband region of interest before KB correction', 
                caption=f'RES = {np.sum(np.abs(self.signal_ft0_ROI - signal_sb_FT_ROI)) / np.sum(np.abs(self.signal_ft0_ROI)):.4f}')

    def kb_correct(self):
        """Apply Koay-Basser correction for both full and probe signals."""
        # Koay-Basser full signal correction

        tot_rician_full = np.sum(self.signal_sb+self.b, axis=0)

        sigma_energy = np.sqrt(tot_rician_full / 2)
        self.sigma = np.repeat(sigma_energy[np.newaxis, :], self.signal_sb_FT.shape[0], axis=0)

        amp_corr = np.zeros_like(self.signal_sb_FT)
        for j in range(self.N_E):
            col_now = np.abs(self.signal_sb_FT[:,j])
            bias_now = tot_rician_full[j]
            amp_corr[:,j] = rk.koay_basser_correction(col_now, bias_now, lambda_thresh=1.0, only_floor = False)
        phase = np.angle(self.signal_sb_FT)

        self.signal_sb_FT = amp_corr * np.exp(1j*phase)

        signal_sb_FT_ROI, em_axis_mid, i0, i1, ef_axis_mid, e0, e1 = rk.extract_midslice(self.signal_sb_FT, self.OMt_slice_fracts, hbar*self.OM_T[:,0], self.Ef_slice_fracts, self.E)
        rk.plot_mat(signal_sb_FT_ROI - 1e-5, extent=[self.x_lo,self.x_hi,self.y_lo*hbar,self.y_hi*hbar],
                saveloc='single_output_temp/pipeline_diag/signal_FT_sb_KB_corr.png', xlabel='Kinetic energy $E_f$ (eV)',ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)', show=False,title='Sideband region of interest after KB correction', 
                caption=f'RES = {np.sum(np.abs(self.signal_ft0_ROI - signal_sb_FT_ROI)) / np.sum(np.abs(self.signal_ft0_ROI)):.4f}')

        self.signal_sb_FT = median_filter(np.real(self.signal_sb_FT),size=(3,3)) + 1j*median_filter(np.imag(self.signal_sb_FT),size=(3,3))

        rk.plot_mat(np.abs(self.signal_ft0_ROI -  signal_sb_FT_ROI ) / np.max(np.abs(self.signal_ft0_ROI)), extent=[self.E_lo,self.E_hi,hbar*self.OM_T[0,0],hbar*self.OM_T[-1,0]],
                saveloc='single_output_temp/pipeline_diag/signal_FT_sb_KB_corr_diff.png', xlabel='Kinetic energy $E_f$ (eV)',ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)', show=False,title='FT of the sideband signal', 
                caption=f'RES = {np.sum(np.abs(self.signal_ft0_ROI - signal_sb_FT_ROI)) / np.sum(np.abs(self.signal_ft0_ROI)):.4f}')

    def probe_reconstruct(self):

        if not hasattr(self, 'rho_params'):
            raise RuntimeError('define_model() must be called before probe_reconstruct() to set fixed rho parameters.')

        sig_power = np.sum(np.abs(self.signal_sb_FT),axis=1)
        dzeta  = 0.1 * np.max(sig_power)

        x_mask = (self.E[0, :] > self.x_lo) & (self.E[0, :] < self.x_hi)
        y_mask =  (sig_power > dzeta) & (np.abs(self.OM_T*hbar)[:,0] < 0.5)

        self.E_zero = self.E[np.ix_(y_mask, x_mask)]
        self.OM_T_zero = self.OM_T[np.ix_(y_mask, x_mask)]
        self.sigma_zero = self.sigma[np.ix_(y_mask, x_mask)]
        self.sigma_zero = np.abs(self.sigma_zero).astype(float)
        self.signal_sb_FT_zero = self.signal_sb_FT[np.ix_(y_mask, x_mask)]

        rk.plot_mat(self.sigma_zero, extent=[self.E_zero[0,0]-self.om_ref*hbar,self.E_zero[0,-1]-self.om_ref*hbar,self.OM_T_zero[1,0]*hbar,self.OM_T_zero[-1,0]*hbar], cmap='plasma',
                 saveloc='aaa1.png', xlabel='Kinetic energy $E_f$ (eV)', ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
                 title='$\\sigma(E_f,\\hbar \\omega_\\tau )$', show=False, mode='abs')

        rk.plot_mat(self.signal_sb_FT_zero - 1e-3, extent=[self.E_zero[0,0]-self.om_ref*hbar,self.E_zero[0,-1]-self.om_ref*hbar,self.OM_T_zero[1,0]*hbar,self.OM_T_zero[-1,0]*hbar], cmap='plasma',
                 saveloc='aaa2.png', xlabel='Kinetic energy $E_f$ (eV)', ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
                 title='$\\tilde S_{corr}(E_f,\\omega_\\tau)$', show=False, square=True)

        ir_lo,xuv_lo,ir_hi,xuv_hi = rk.conv_bounds(self.x_lo/hbar,self.x_hi/hbar,self.N_E,self.om_ref)
        om_probe_zero = np.linspace(ir_lo,ir_hi,self.E_zero.shape[1])
        om_xuv_zero = np.linspace(xuv_lo,xuv_hi,self.E_zero.shape[1])
        OM_P_zero = np.tile(om_probe_zero, (self.E_zero.shape[0], 1))
        OM_X_zero = np.tile(om_xuv_zero, (self.E_zero.shape[0], 1))

        # # Build full frequency grids used by the zero-omega forward model and apply the same ROI masks.
        # OM_P = np.tile(self.om_probe, (self.N_T, 1))
        # OM_X = np.tile(self.om_xuv, (self.N_T, 1))
        # OM_P_zero = OM_P[np.ix_(y_mask, x_mask)]
        # OM_X_zero = OM_X[np.ix_(y_mask, x_mask)]

        from MCMCprobe import Bayesian_MCMC as Bayesian_MCMC_probe

        probe_amp_hat, probe_mean_hat, probe_sigma_hat, z_fit_zero = Bayesian_MCMC_probe(
            om_p=OM_P_zero,
            om_x=OM_X_zero,
            om_t=self.OM_T_zero,
            z_obs=self.signal_sb_FT_zero,
            sigma_obs=self.sigma_zero,
            rho_params=self.rho_params,
            om_ref=self.om_ref,
            hbar=hbar,
            num_warmup=1000,
            num_samples=2000,
            num_chains=1,
            rng_seed=0,
            save_prefix='single_output_temp/pipeline_diag/probe_mcmc',
        )

        self.probe_amp_hat = probe_amp_hat
        self.probe_mean_hat = probe_mean_hat
        self.probe_sigma_hat = probe_sigma_hat
        self.signal_sb_FT_zero_fit = z_fit_zero

        # Reconstruct 1D probe spectrum from inferred single-Gaussian parameters.
        sigma_eff = max(self.probe_sigma_hat, 1e-12)
        self.sp_probe_inferred = self.probe_amp_hat * np.exp(
            -0.5 * ((self.om_probe - self.probe_mean_hat) / sigma_eff) ** 2
        )

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(self.om_probe*hbar, rk.normalize_abs(np.abs(self.sp_probe)), label='True probe', linewidth=2)
        ax.plot(self.om_probe*hbar, rk.normalize_abs(self.sp_probe_inferred), '--', label='MCMC inferred probe', linewidth=2)
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('Amplitude [arb. u.]')
        ax.set_title('Probe spectrum: true vs MCMC inferred')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig('single_output_temp/pipeline_diag/probe_spectrum_fit.png', dpi=300)
        plt.close(fig)

        rk.plot_mat(
            np.abs(self.signal_sb_FT_zero - self.signal_sb_FT_zero_fit)/np.max(np.abs(self.signal_sb_FT_zero_fit)),
            extent=[self.E_zero[0,0]-self.om_ref*hbar, self.E_zero[0,-1]-self.om_ref*hbar, self.OM_T_zero[1,0]*hbar, self.OM_T_zero[-1,0]*hbar],
            cmap='plasma',
            saveloc='single_output_temp/pipeline_diag/probe_zeroomega_abs_residual.png',
            xlabel='Kinetic energy $E_f$ (eV)',
            ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
            title='Zero-omega fit residual |data - model|',
            show=False,
            mode='abs',
        )

        print(
            'Inferred probe parameters (single Gaussian):',
            f'amp={self.probe_amp_hat:.6g},',
            f'mean={self.probe_mean_hat:.6g},',
            f'sigma={self.probe_sigma_hat:.6g}'
        )

if __name__ == "__main__":

    E_lo = 24.5
    E_hi = 28.0
    T_reach = 150
    E_res = 0.01    
    N_T = 501
    p_E = 4  # N_E upsampling integer
    alpha = 10000
    b = 1

    sideband_lo = 25.5
    sideband_hi = 28.0
    harmq_lo = 24.5
    harmq_hi = 25.5

    if_coherent = True

    # Define pulses

    # A_probe = 0.6
    # a_probes = [1.0,0.05,0.05]
    # om_probes = [1.55/hbar,1.52/hbar,1.58/hbar]
    # s_probes = [0.15/hbar,0.02/hbar,0.02/hbar]

    A_probe = 0.6
    a_probes = [1.0]
    om_probes = [1.55/hbar]
    s_probes = [0.08/hbar]

    # A_probe = 0.6  # WIDE PROBE VARIANT
    # a_probes = [1.0]
    # om_probes = [1.55/hbar]
    # s_probes = [0.18/hbar]

    A_ref = 1.0
    om_ref = 1.50/hbar
    s_ref = 0.025/hbar

    # A_ref = 1.0  # WIDE PROBE VARIANT
    # om_ref = 1.55/hbar
    # s_ref = 0.025/hbar

    # Create experiment instance
    experiment = RK_experiment(E_lo=E_lo,E_hi=E_hi,T_reach=T_reach,E_res=E_res,N_T=N_T,p_E=p_E,alpha=alpha,b=b,
                            sb_lo=sideband_lo,sb_hi=sideband_hi,harmq_lo=harmq_lo,harmq_hi=harmq_hi,if_coherent=if_coherent,
                            A_ref=A_ref,om_ref=om_ref,s_ref=s_ref)

    experiment.ifWF = False
    experiment.ifWide = False

    experiment.define_pulses(A_probe,a_probes,om_probes,s_probes)
    experiment.define_model()
    experiment.generate_signal()
    experiment.process_and_detrend()
    experiment.kb_correct()
    experiment.probe_reconstruct()