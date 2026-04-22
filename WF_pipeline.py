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
import rkraken as rk
from MCMCrho import Bayesian_MCMC

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
    probe_abs = np.abs(sp_pr)
    ref_abs = np.abs(sp_ref)
    line_probe_abs, = ax1.plot(om_pr * hbar, probe_abs, label='Probe |spectrum|', linewidth=2)
    line_ref_abs, = ax1.plot(om_pr * hbar, ref_abs, label='Reference |spectrum|', linewidth=2)

    # Overlay phase on a second y-axis; hide low-SNR regions below 5% of each peak.
    ax1_phase = ax1.twinx()
    probe_phase = np.angle(sp_pr)
    ref_phase = np.angle(sp_ref)
    probe_phase_mask = probe_abs >= 0.05 * np.max(probe_abs)
    ref_phase_mask = ref_abs >= 0.05 * np.max(ref_abs)
    probe_phase_plot = np.where(probe_phase_mask, probe_phase, np.nan)
    ref_phase_plot = np.where(ref_phase_mask, ref_phase, np.nan)
    line_probe_phase, = ax1_phase.plot(
        om_pr * hbar,
        probe_phase_plot,
        '--',
        linewidth=1.2,
        alpha=0.8,
        label='Probe phase',
        color='tab:green',
    )
    line_ref_phase, = ax1_phase.plot(
        om_pr * hbar,
        ref_phase_plot,
        '--',
        linewidth=1.2,
        alpha=0.8,
        label='Reference phase',
        color='tab:red',
    )

    ax1.set_xlabel('Energy [eV]',fontweight='bold')
    ax1.set_ylabel('Amplitude [arb. u.]',fontweight='bold')
    ax1_phase.set_ylabel('Phase [rad]', fontweight='bold')
    ax1_phase.set_ylim([-np.pi, np.pi])
    ax1_phase.set_yticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
    ax1.set_title('Probe and Reference Spectra',fontweight='bold',fontsize=12)
    ax1.set_xlim([0.8,2.5])

    lines = [line_probe_abs, line_ref_abs, line_probe_phase, line_ref_phase]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best')
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

    def define_pulses(self, probe_params):
        self.probe_params = probe_params
        # # Define XUV pulse configurations (stationary, tau=0)
        # xuvs_list = []
        # for i in range(len(a_xuvs)):
        #     xuv_pulse = (0*self.T, A_xuv*a_xuvs[i], om_xuvs[i], s_xuvs[i], 0)
        #     xuvs_list.append(xuv_pulse)
        # self.xuvs = tuple(xuvs_list)
        
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

        # amps = [1.0/np.sqrt(2),1.0]
        # mus = [25.0-0.20+self.om_ref*hbar,25.0+self.om_ref*hbar]
        # sigmas = [0.08,0.08]
        # betas = [2,2]
        # taus = [1,1]
        # lambdas = [3,7]
        # gammas = np.array([[1.0,0.5],
        #                    [0.5,1.0]])
        # etas = np.array([[1.0,0.0],
        #                  [0.0,1.0]])
        
        # amps = [1.0,1.0,1.0]
        # mus = [25.0-0.30+self.om_ref*hbar,25.0+self.om_ref*hbar,25.0+0.30+self.om_ref*hbar]
        # sigmas = [0.06,0.06,0.06]
        # betas = [0,0,0]
        # taus = [0,0,0]
        # lambdas = [5,15,10]
        # gammas = np.array([[1.0,1.0,1.0],
        #                    [1.0,1.0,1.0],
        #                    [1.0,1.0,1.0]])
        # etas = np.array([[1.0,-1.0,-1.0],
        #                  [-1.0,1.0,-1.0],
        #                  [-1.0,-1.0,1.0]])

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

        self.sp_probe = self.sp_tot( self.om_probe)
        self.sp_ref = self.sp_probe*ref_mask
        # self.sp_xuv = rk.sp_tot(self.xuvs, self.om_xuv)

        # ir_lo,xuv_lo,ir_hi,xuv_hi = rk.conv_bounds(self.E_lo/hbar,self.E_hi/hbar,self.N_E_up,self.om_ref)
        # self.om_probe_up = np.linspace(ir_lo,ir_hi,self.N_E_up)
        # self.om_xuv_up = np.linspace(xuv_lo,xuv_hi,self.N_E_up)

        # ref_mask_up = self.A_ref*((self.om_probe_up >= self.om_ref - self.s_ref) & (self.om_probe_up <= self.om_ref + self.s_ref)).astype(float)        

        # self.sp_probe_up = self.sp_tot( self.om_probe_up)
        # self.sp_ref_up = self.sp_probe_up*ref_mask_up
        # self.sp_xuv_up = rk.sp_tot(self.xuvs, self.om_xuv_up)

        # om_probe_reg = rk.regularize_omega(self.om_probe_up)
        # om_xuv_reg = rk.regularize_omega(self.om_xuv_up)

        ### PLOT the upsampled spectra with probe/ref on one axis, xuv on another

        plot_spectra(self.om_probe,self.om_xuv,self.sp_probe,self.sp_ref,np.abs(self.rho_f(self.om_xuv*hbar + self.om_ref*hbar,self.om_xuv*hbar + self.om_ref*hbar)))
        
        ### SIMULATE SIGNAL

        _, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, np.zeros((self.N_T,self.N_E)), use_window=False, zero_pad=self.zero_pad)

        OM_P = np.tile(self.om_probe, (self.N_T, 1))
        OM_X = np.tile(self.om_xuv, (self.N_T, 1))

        a_ref = self.A_ref * self.sp_tot(self.om_ref) / self.om_ref
        a_pr = self.sp_tot(self.OM_T) / rk.regularize_omega(self.OM_T)

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
        in_2_num = (self.sp_tot( OM_P + self.OM_T)
            * np.conj(self.sp_tot(OM_P)))
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

        # if self.if_coherent:
        #     sp_xuv_E = rk.sp_tot(self.xuvs, self.E_up/hbar)
        #     # Second-order amplitudes
        #     A2_probe = rk.synth_baseline_n(self.sp_probe_up, self.sp_xuv_up, 
        #                                  om_probe_reg, om_xuv_reg, self.T_up)
        #     A2_ref = rk.synth_baseline_n(self.sp_ref_up, self.sp_xuv_up, 
        #                                om_probe_reg, om_xuv_reg, self.T_up*0)

        #     A2_total = A2_probe + A2_ref

        #     amplit_tot_0 = rk.downsample(
        #                             A2_total
        #                             + sp_xuv_E
        #                             , self.p_E)
        #     signal_clean = np.abs(amplit_tot_0)**2

        #     self.signal_zerocomp = np.abs(rk.downsample(A2_probe, self.p_E))**2 + np.abs(rk.downsample(A2_ref, self.p_E))**2

        # else:
        #     # Incoherent: sum |amplitude|^2 from each XUV component
        #     signal_clean = np.zeros((self.N_T, self.N_E))
        #     for xuv_i in self.xuvs:
        #         sp_xuv_i = rk.sp_tot((xuv_i,), self.om_xuv_up)
        #         sp_xuv_i_E = rk.sp_tot((xuv_i,), self.E_up/hbar)
        #         # Second-order amplitudes
        #         A2_probe_i = rk.synth_baseline_n(self.sp_probe_up, sp_xuv_i, 
        #                                        om_probe_reg, om_xuv_reg, self.T_up)
        #         A2_ref_i = rk.synth_baseline_n(self.sp_ref_up, sp_xuv_i, 
        #                                      om_probe_reg, om_xuv_reg, self.T_up*0)
                
        #         A2_total_i = A2_probe_i + A2_ref_i

        #         amplit_i = rk.downsample(
        #                             A2_total_i
        #                             + sp_xuv_i_E
        #                             , self.p_E)
        #         signal_clean += np.abs(amplit_i)**2
        

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

        # self.signal_sb_FT = median_filter(np.real(self.signal_sb_FT),size=(3,3)) + 1j*median_filter(np.imag(self.signal_sb_FT),size=(3,3))

        # signal_sb_FT_ROI, em_axis_mid, i0, i1, ef_axis_mid, e0, e1 = rk.extract_midslice(self.signal_sb_FT, self.OMt_slice_fracts, hbar*self.OM_T[:,0], self.Ef_slice_fracts, self.E)
        # rk.plot_mat(signal_sb_FT_ROI - 1e-5, extent=[self.x_lo,self.x_hi,self.y_lo*hbar,self.y_hi*hbar],
        #         saveloc='single_output_temp/pipeline_diag/signal_FT_sb_KB_corr_median.png', xlabel='Kinetic energy $E_f$ (eV)',ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)', show=False,title='FT of the sideband signal', 
        #         caption=f'RES = {np.sum(np.abs(self.signal_ft0_ROI - signal_sb_FT_ROI)) / np.sum(np.abs(self.signal_ft0_ROI)):.4f}')

        rk.plot_mat(np.abs(self.signal_ft0_ROI -  signal_sb_FT_ROI ) / np.max(np.abs(self.signal_ft0_ROI)), extent=[self.E_lo,self.E_hi,hbar*self.OM_T[0,0],hbar*self.OM_T[-1,0]],
                saveloc='single_output_temp/pipeline_diag/signal_FT_sb_KB_corr_diff.png', xlabel='Kinetic energy $E_f$ (eV)',ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)', show=False,title='FT of the sideband signal', 
                caption=f'RES = {np.sum(np.abs(self.signal_ft0_ROI - signal_sb_FT_ROI)) / np.sum(np.abs(self.signal_ft0_ROI)):.4f}')

    def probe_sp_correct(self):
        
        if self.ifWide:
            dzeta = 0.1 * np.max(np.abs(self.sp_tot(self.OM_T))) # WIDE PROBE VARIANT
        else:
            dzeta = 0.05 * np.max(np.abs(self.sp_tot(self.OM_T)))

        probe_modulation = self.sp_tot(self.OM_T) / np.maximum(self.OM_T,0.01)

        where_off = probe_modulation < dzeta
        probe_modulation[where_off] = dzeta

        self.signal_sb_FT_corrected = self.signal_sb_FT / probe_modulation
        self.signal_sb_FT_corrected[where_off] = 0

        self.sigma = self.sigma / np.abs(probe_modulation)
        self.sigma[where_off] = 0

        x_mask = (self.E[0, :] > self.x_lo) & (self.E[0, :] < self.x_hi)
        y_mask = probe_modulation[:,0] > dzeta

        if self.ifWide:
            self.signal_sb_FT_corrected = median_filter(np.real(self.signal_sb_FT_corrected),size=(3,3)) + 1j*median_filter(np.imag(self.signal_sb_FT_corrected),size=(3,3))  # WIDE PROBE VARIANT

        self.E_rho = self.E[np.ix_(y_mask, x_mask)]
        self.OM_T_rho = self.OM_T[np.ix_(y_mask, x_mask)]
        self.sigma_rho = self.sigma[np.ix_(y_mask, x_mask)]
        self.signal_sb_FT_corrected_rho = self.signal_sb_FT_corrected[np.ix_(y_mask, x_mask)]

        self.sigma_rho = np.abs(self.sigma_rho).astype(float)

        rk.plot_mat(self.sigma_rho, extent=[self.E_rho[0,0]-self.om_ref*hbar,self.E_rho[0,-1]-self.om_ref*hbar,self.OM_T_rho[1,0]*hbar,self.OM_T_rho[-1,0]*hbar], cmap='plasma',
                 saveloc='single_output_temp/rhos/sigmas.png', xlabel='Kinetic energy $E_f$ (eV)', ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
                 title='$\\sigma(E_f,\\hbar \\omega_\\tau )$', show=False, mode='abs')

        rk.plot_mat(self.signal_sb_FT_corrected_rho - 1e-3, extent=[self.E_rho[0,0]-self.om_ref*hbar,self.E_rho[0,-1]-self.om_ref*hbar,self.OM_T_rho[1,0]*hbar,self.OM_T_rho[-1,0]*hbar], cmap='plasma',
                 saveloc='single_output_temp/rhos/rho_rec_unproj.png', xlabel='Kinetic energy $E_f$ (eV)', ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
                 title='$\\tilde S_{corr}(E_f,\\omega_\\tau)$', show=False, square=True)

        return

    def resample_analyze(self):
        """Resample and analyze results to produce final density matrices."""
        # Resample and analyze

        E1 = self.E_rho - self.OM_T_rho*hbar + self.om_ref*hbar - self.om_ref*hbar
        E2 = self.E_rho - self.om_ref*hbar

        N_NEW = 100
        rho_raw, amplit_tot_FT_corrected_small, extent_small, idxs_small, E1interp, E2interp = rk.resample(
            self.signal_sb_FT_corrected, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, N_NEW)
        rho_raw_sigma, amplit_tot_FT_corrected_small, extent_small, idxs_small, E1interp, E2interp = rk.resample(
            self.sigma, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, N_NEW)

        ideal_rho = self.rho_f(E1interp + self.om_ref*hbar, E2interp + self.om_ref*hbar)
        ideal_rho = ideal_rho/np.trace(ideal_rho)

        raw_trace = np.trace(rho_raw)
        rho_raw = rho_raw/raw_trace
        rho_raw_sigma = rho_raw_sigma/raw_trace

        rk.plot_mat(rho_raw - 1e-4, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 saveloc='single_output_temp/rhos/rho_raw', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
                 title='$\\tilde S_{corr}(\\varepsilon_2,\\varepsilon_1)$', show=False, square=True)
        
        rk.plot_mat(rho_raw_sigma, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 saveloc='single_output_temp/rhos/rho_raw_sigma', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
                 title='$\\sigma (\\varepsilon_2,\\varepsilon_1)$', show=False, mode='abs')

        rk.plot_mat(ideal_rho - 1e-4, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 saveloc='single_output_temp/rhos/rho_ideal.png', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
                 title='Initial photoelectron density matrix', show=False, square=True)
        
        if self.ifWide:
            inferred_rho = rk.project_to_density_matrix(rho_raw,4)
            fid0 = rk.fidelity(ideal_rho, inferred_rho)

            rk.plot_mat(inferred_rho - 1e-4, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                    saveloc='single_output_temp/rhos/rho_inferred.png', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
                    title='Inferred density matrix', show=False, caption='F=%.3f'%fid0, square=True)

            rk.plot_mat((ideal_rho - inferred_rho) / np.max(np.abs(ideal_rho)), extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                    saveloc='single_output_temp/rhos/rho_diff.png', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
                    title='$(\\rho_0 - \\rho_{\\text{inferred}}) / \\text{max}(|\\rho_0|) $', show=False, square=True)
            
        else:
            selected_indices = self.sigma_rho.flatten() > 0

            amps_hat, mus_hat, sigmas_hat, betas_hat, taus_hat, lambdas_hat, gamma_hat, eta_hat = Bayesian_MCMC(E1.flatten()[selected_indices],E2.flatten()[selected_indices],
                        self.signal_sb_FT_corrected_rho.flatten()[selected_indices],self.sigma_rho.flatten()[selected_indices],n_peaks=2)

            def rho_inferred(e1,e2):
                return rho_model(e1,e2,amps_hat,mus_hat,sigmas_hat,betas_hat,taus_hat,lambdas_hat,gamma_hat,eta_hat)

            inferred_rho = rho_inferred(E1interp, E2interp)
            inferred_rho = inferred_rho/np.trace(inferred_rho)

            fid0 = rk.fidelity(ideal_rho, inferred_rho)

            rk.plot_mat(inferred_rho - 1e-4, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                    saveloc='single_output_temp/rhos/rho_inferred.png', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
                    title='Inferred density matrix', show=False, caption='F=%.3f'%fid0, square=True)

            rk.plot_mat((ideal_rho - inferred_rho) / np.max(np.abs(ideal_rho)), extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                    saveloc='single_output_temp/rhos/rho_diff.png', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
                    title='$(\\rho_0 - \\rho_{\\text{inferred}}) / \\text{max}(|\\rho_0|) $', show=False, square=True)

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
    experiment.probe_sp_correct()
    experiment.resample_analyze()