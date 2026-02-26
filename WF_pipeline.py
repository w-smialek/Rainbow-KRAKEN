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
from math import floor
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from matplotlib import patheffects as pe
from skimage.restoration import denoise_tv_bregman
from scipy.ndimage import median_filter
from scipy.special import i0e, i1e
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import sqrtm
import rkraken as rk

# Reduced Planck constant in eV*fs (approx CODATA): ħ = 6.582119569e-16 eV·s
hbar = 6.582119569e-1

class RK_experiment:
    """Rainbow-KRAKEN experiment class for full pipeline processing."""
    
    def __init__(self,E_lo=60.0,E_hi=63.5,T_reach=50,E_res=0.025,N_T=240,p_E=4,alpha=0.05,b=1,sb_lo=24.7,sb_hi=28,harmq_lo=22,if_coherent=True,E_range=None):
        # Field parameters
        self.E_lo = E_lo
        self.E_hi = E_hi
        self.T_reach = T_reach
        self.E_span = self.E_hi - self.E_lo
        
        self.E_res = E_res
        self.N_E = round(self.E_span/self.E_res/10)*10
        print(self.N_E)
        
        self.N_T = N_T
        self.p_E = p_E  # N_E upsampling integer
        
        self.alpha = alpha
        self.b = b
        
        # Coherence flag
        self.if_coherent = if_coherent
        
        # Noise and processing parameters
        self.ifnoise = True
        # self.noise_area_Elo = 0.0  NOISE FLOOR SHOULD BE QUITE PRECISELY MEASURED BY CAPTURING WITH LASER OFF
        # self.noise_area_Ehi = 0.1
        self.sb_lo = sb_lo
        self.sb_hi = sb_hi
        self.harmq_lo = harmq_lo
        self.zero_pad = 0
        
        # Detrending parameters
        self.em_axis_mid_reach = 1.00
        self.T_mix_reach = 40
        self.mask_lo = 1.35   # indirect-energy lower edge of the feature band (eV)
        self.mask_hi = 1.75   # indirect-energy upper edge of the feature band (eV)
        self.donor_lo = 0.60  # lower edge of the feature-free donor band (eV)
        
        # Correction parameters
        self.dzeta_val = 1e-3
        self.theta_val = 0.01
        
        # Resampling parameters
        self.rho_lo = harmq_lo
        self.rho_hi = sb_lo
        
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

    def define_pulses(self,A_xuv,A_probe,A_ref,a_xuvs,a_probes,a_refs,om_xuvs,om_probes,om_refs,s_xuvs,s_probes,s_refs):
        # Define XUV pulse configurations (stationary, tau=0)
        xuvs_list = []
        for i in range(len(a_xuvs)):
            xuv_pulse = (0*self.T, A_xuv*a_xuvs[i], om_xuvs[i], s_xuvs[i], 0)
            xuvs_list.append(xuv_pulse)
        self.xuvs = tuple(xuvs_list)
        
        # Define probe pulse configurations (time-dependent, tau=T)
        probes_list = []
        for i in range(len(a_probes)):
            probe_pulse = (self.T, A_probe*a_probes[i], om_probes[i], s_probes[i], 0)
            probes_list.append(probe_pulse)
        self.probes = tuple(probes_list)
        
        # Define reference pulse configurations (stationary, tau=0)
        refs_list = []
        for i in range(len(a_refs)):
            ref_pulse = (0*self.T, A_ref*a_refs[i], om_refs[i], s_refs[i], 0)
            refs_list.append(ref_pulse)
        self.refs = tuple(refs_list)
        
        # Store reference frequency for later use
        self.om_ref = om_refs[0]  # Use first reference frequency
        
        # Store XUV parameters for ground truth comparison
        self.om0_xuv = om_xuvs[0]
        self.s_xuv = s_xuvs[0]
        
        self.refprobes = tuple(list(self.refs) + list(self.probes))
    
    def generate_signal(self,exp_signal=None):
        """Generate synthetic baseline and full signal with optional noise."""
        # Synthetic baseline with known spectra
        self.om_probe = (self.E/hbar - ((self.E_lo/hbar+self.E_hi/hbar)/2-self.om_ref) + self.E_span/hbar/self.N_E/2 * ((self.N_E-1) % 2))[0,:]
        self.om_xuv = (self.E/hbar - self.om_ref)[0,:]
        
        self.sp_xuv = rk.sp_tot(self.xuvs, self.om_xuv)

        plt.plot(self.om_xuv*hbar,self.sp_xuv)
        plt.savefig('figg.png')
        plt.close()

        self.sp_probe = rk.sp_tot(self.probes, self.om_probe) #* np.exp(1j*(np.linspace(-np.pi,np.pi,np.size(self.om_probe)))**2)
        self.sp_ref = rk.sp_tot(self.refs, self.om_probe)
        
        self.om_probe_up = (self.E_up/hbar - ((self.E_lo/hbar+self.E_hi/hbar)/2-self.om_ref) + self.E_span/hbar/self.N_E_up/2 * ((self.N_E_up-1) % 2))[0,:]
        self.om_xuv_up = (self.E_up/hbar - self.om_ref)[0,:]

        self.sp_xuv_up = rk.sp_tot(self.xuvs, self.om_xuv_up)
        self.sp_probe_up = rk.sp_tot(self.probes, self.om_probe_up) #* np.exp(1j*(np.linspace(-np.pi,np.pi,np.size(self.om_probe_up)))**2)
        self.sp_ref_up = rk.sp_tot(self.refs, self.om_probe_up)
        
        om_probe_up_emit = (self.E_up/hbar - ((self.E_lo/hbar+self.E_hi/hbar)/2+self.om_ref) + self.E_span/hbar/self.N_E_up/2 * ((self.N_E_up-1) % 2))[0,:]
        om_xuv_up_emit = (self.E_up/hbar + self.om_ref)[0,:]
        
        sp_xuv_up_emit = rk.sp_tot(self.xuvs, om_xuv_up_emit)
        sp_probe_up_emit = rk.sp_tot(rk.pulse_emit(self.probes), om_probe_up_emit)
        sp_ref_up_emit = rk.sp_tot(rk.pulse_emit(self.refs), om_probe_up_emit)

        # Plot the upsampled spectra with probe/ref on one axis, xuv on another
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Top subplot: Probe and Reference spectra
        ax1.plot(self.om_probe_up * hbar, self.sp_probe_up, label='Probe spectrum', linewidth=2)
        ax1.plot(self.om_probe_up * hbar, self.sp_ref_up, label='Reference spectrum', linewidth=2)
        ax1.set_xlabel('Energy [eV]',fontweight='bold')
        ax1.set_ylabel('Amplitude [arb. u.]',fontweight='bold')
        ax1.set_title('Probe and Reference Spectra',fontweight='bold',fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot: XUV spectrum
        ax2.plot(self.om_xuv_up * hbar, self.sp_xuv_up, label='XUV spectrum', linewidth=2, color='purple')
        ax2.set_xlabel('Energy [eV]',fontweight='bold')
        ax2.set_ylabel('Amplitude [arb. u.]',fontweight='bold')
        ax2.set_title('XUV Spectrum',fontweight='bold',fontsize=12)
        # ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('single_output_temp/spectra/input_spectra.png', dpi=300)
        plt.close()
        
        om_probe_reg = rk.regularize_omega(self.om_probe_up)
        om_xuv_reg = rk.regularize_omega(self.om_xuv_up)
        
        if exp_signal is None:

            # Probe only for reference
            if self.if_coherent:
                self.synth_bsln = np.abs(rk.downsample(rk.synth_baseline(self.sp_probe_up, self.sp_xuv_up, 
                                                                om_probe_reg, om_xuv_reg, self.T_up), self.p_E))**2
            else:
                # Incoherent: sum intensities from each XUV component separately
                self.synth_bsln = np.zeros((self.N_T, self.N_E))
                for xuv_i in self.xuvs:
                    sp_xuv_i = rk.sp_tot((xuv_i,), self.om_xuv_up)
                    self.synth_bsln += np.abs(rk.downsample(rk.synth_baseline(self.sp_probe_up, sp_xuv_i, 
                                                                om_probe_reg, om_xuv_reg, self.T_up), self.p_E))**2
            
            self.synth_bsln = self.rng.poisson(self.alpha*(self.synth_bsln)+self.b).astype(float) - self.b
            self.synth_bsln_FT, _, el, eh = rk.CFT(self.T_range, self.synth_bsln, use_window=False)
            
        
            # Synthetic full signal
            if self.if_coherent:
                sp_xuv_E = rk.sp_tot(self.xuvs, self.E_up/hbar)
                amplit_tot_0 = rk.downsample(
                                        rk.synth_baseline(self.sp_probe_up, self.sp_xuv_up, 
                                                    om_probe_reg, om_xuv_reg, self.T_up)
                                        + rk.synth_baseline(self.sp_ref_up, self.sp_xuv_up, 
                                                    om_probe_reg, om_xuv_reg, self.T_up*0)
                                        + sp_xuv_E
                                        , self.p_E)
                signal_clean = np.abs(amplit_tot_0)**2
            else:
                # Incoherent: sum |amplitude|^2 from each XUV component
                signal_clean = np.zeros((self.N_T, self.N_E))
                for xuv_i in self.xuvs:
                    sp_xuv_i = rk.sp_tot((xuv_i,), self.om_xuv_up)
                    sp_xuv_i_E = rk.sp_tot((xuv_i,), self.E_up/hbar)
                    amplit_i = rk.downsample(
                                        rk.synth_baseline(self.sp_probe_up, sp_xuv_i, 
                                                    om_probe_reg, om_xuv_reg, self.T_up)
                                        + rk.synth_baseline(self.sp_ref_up, sp_xuv_i, 
                                                    om_probe_reg, om_xuv_reg, self.T_up*0)
                                        + sp_xuv_i_E
                                        , self.p_E)
                    signal_clean += np.abs(amplit_i)**2
            
            signal_clean *= self.alpha
            signal_clean += self.b
            
            if not self.ifnoise:
                self.signal = signal_clean.copy()
            else:
                # Apply Poisson noise assuming signal_clean contains expected values (means)
                self.signal = self.rng.poisson(signal_clean).astype(float)
            
            print("N Total = %.2e"%np.sum(self.signal))
        
            self.signal -= self.b # NOISE FLOOR SHOULD BE QUITE PRECISELY MEASURED BY CAPTURING WITH LASER OFF

            # Zero out signal outside the sideband range [sb_lo, sb_hi]
            sb_lo_idx = np.argmin(np.abs(self.E_range - self.sb_lo))
            sb_hi_idx = np.argmin(np.abs(self.E_range - self.sb_hi))
            self.signal_sb = np.zeros_like(self.signal)
            self.signal_sb[:, sb_lo_idx:sb_hi_idx+1] = self.signal[:, sb_lo_idx:sb_hi_idx+1]

            harmq_lo_idx = np.argmin(np.abs(self.E_range - self.harmq_lo))
            self.signal_harmq = np.zeros_like(self.signal)
            self.signal_harmq[:, harmq_lo_idx:sb_lo_idx+1] = self.signal[:, harmq_lo_idx:sb_lo_idx+1]

            peak_sb_counts = int(np.max(self.signal_sb))
            self.peak_sb_counts = peak_sb_counts

            self.xuv_peak()

        else:
            self.signal = exp_signal

            # Zero out signal outside the sideband range [sb_lo, sb_hi]
            sb_lo_idx = np.argmin(np.abs(self.E_range - self.sb_lo))
            sb_hi_idx = np.argmin(np.abs(self.E_range - self.sb_hi))
            self.signal_sb = np.zeros_like(self.signal)
            self.signal_sb[:, sb_lo_idx:sb_hi_idx+1] = self.signal[:, sb_lo_idx:sb_hi_idx+1]

            harmq_lo_idx = np.argmin(np.abs(self.E_range - self.harmq_lo))
            self.signal_harmq = np.zeros_like(self.signal)
            self.signal_harmq[:, harmq_lo_idx:sb_lo_idx+1] = self.signal[:, harmq_lo_idx:sb_lo_idx+1]

            peak_sb_counts = int(np.max(self.signal_sb))
            self.peak_sb_counts = peak_sb_counts

            self.xuv_peak()

            # self.xuvs = self.xuvs_rec

            # Probe only for reference
            if self.if_coherent:
                self.synth_bsln = np.abs(rk.downsample(rk.synth_baseline(self.sp_probe_up, rk.sp_tot(self.xuv, self.om_xuv_up), 
                                                                om_probe_reg, om_xuv_reg, self.T_up), self.p_E))**2
            else:
                # Incoherent: sum intensities from each XUV component separately
                self.synth_bsln = np.zeros((self.N_T, self.N_E))
                for xuv_i in self.xuvs:
                    sp_xuv_i = rk.sp_tot((xuv_i,), self.om_xuv_up)
                    self.synth_bsln += np.abs(rk.downsample(rk.synth_baseline(self.sp_probe_up, sp_xuv_i, 
                                                                om_probe_reg, om_xuv_reg, self.T_up), self.p_E))**2
            
            self.synth_bsln = self.rng.poisson(self.alpha*(self.synth_bsln)+self.b).astype(float) - self.b
            self.synth_bsln_FT, _, el, eh = rk.CFT(self.T_range, self.synth_bsln, use_window=False)
            
    
    def process_and_detrend(self):
        """Process signal, estimate background, and perform detrending."""
        # Processing starts here
        # self.b_est = np.mean(self.signal[:,floor(self.noise_area_Elo*self.N_E):floor(self.noise_area_Ehi*self.N_E)])
        # self.signal -= self.b_est



        # self.signal = self.signal_sb
        

        
        rk.plot_mat(self.signal, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
                caption='Peak sb counts: %i\nN_T: %i\nN_E: %i\nT_res: %.2f fs\nE_res: %.3f eV'%(
                    self.peak_sb_counts,self.N_T,self.N_E,2*self.T_reach/self.N_T,self.E_res),
                saveloc='single_output_temp/pipeline_diag/measured_signal.png',xlabel='Kinetic energy (eV)',ylabel='Time (fs)')
        
        rk.plot_mat(self.signal_sb, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
                # caption='Peak sb counts: %i\nN_T: %i\nN_E: %i\nT_res: %.2f fs\nE_res: %.3f eV'%(
                #     self.peak_sb_counts,self.N_T,self.N_E,2*self.T_reach/self.N_T,self.E_res),
                saveloc='single_output_temp/pipeline_diag/measured_signal_sb.png',xlabel='Kinetic energy (eV)',ylabel='Time (fs)',title='Sideband signal')
        
        rk.plot_mat(self.signal_harmq, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
                caption='Peak sb counts: %i\nN_T: %i\nN_E: %i\nT_res: %.2f fs\nE_res: %.3f eV'%(
                    self.peak_sb_counts,self.N_T,self.N_E,2*self.T_reach/self.N_T,self.E_res),
                saveloc='single_output_temp/pipeline_diag/measured_signal_harmq.png',xlabel='Kinetic energy (eV)',ylabel='Time (fs)')
        
        self.amplit_tot_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal_sb, use_window=True, zero_pad=self.zero_pad)
        self.amplit_tot_FT_wndw, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal_sb, use_window=True, zero_pad=self.zero_pad)
        
        rk.plot_mat(self.amplit_tot_FT_wndw, extent=[self.E_lo,self.E_hi,em_lo,em_hi],
                saveloc='single_output_temp/pipeline_diag/raw_signal_FT.png', show=False)
        
        ROI_reach = 2.3
        slice_fracts = (0.5*(1 - ROI_reach/em_hi), 0.5*(1 + ROI_reach/em_hi))
        
        amplit_tot_FT_ROI, em_axis_mid, i0, i1 = rk.extract_midslice(self.amplit_tot_FT, slice_fracts, hbar*self.OM_T[:,0])
        
        rk.plot_mat(np.minimum(amplit_tot_FT_ROI,4*self.peak_sb_counts), extent=[self.E_lo,self.E_hi,-ROI_reach,ROI_reach],
                saveloc='single_output_temp/pipeline_diag/raw_signal_FT_ROI.png',xlabel='Kinetic energy (eV)',ylabel='Indirect energy (eV)', show=False,title='FT of the sideband signal')
        
        ROI_reach_lo = 1.0
        ROI_reach_hi = 2.2
        slice_fracts = (0.5*(1 + ROI_reach_lo/em_hi), 0.5*(1 + ROI_reach_hi/em_hi))
        ROI_e_reach_lo = self.sb_lo
        ROI_e_reach_hi = self.sb_hi
        slice_e_fracts = ((ROI_e_reach_lo - self.E_lo) / (self.E_hi - self.E_lo), 
                          (ROI_e_reach_hi - self.E_lo) / (self.E_hi - self.E_lo))
        
        amplit_tot_FT_ROI, em_axis_mid, i0, i1, e_axis_mid, e_i0, e_i1 = rk.extract_midslice(self.amplit_tot_FT, slice_fracts, hbar*self.OM_T[:,0],
                                                                                             e_slice_fracts=slice_e_fracts, e_sliced_range=self.E[0,:])
        
        rk.plot_mat(np.minimum(amplit_tot_FT_ROI,4*self.peak_sb_counts), extent=[ROI_e_reach_lo,ROI_e_reach_hi,ROI_reach_lo,ROI_reach_hi],
                saveloc='single_output_temp/pipeline_diag/raw_signal_FT_ROI2.png', show=False)
        
        
        # Detrending - separating probe and ref zero freq component
        slice_fracts = (0.5*(1 - self.em_axis_mid_reach/em_hi), 0.5*(1 + self.em_axis_mid_reach/em_hi))
        
        amplit_tot_FT_mid, em_axis_mid, i0, i1 = rk.extract_midslice(self.amplit_tot_FT, slice_fracts, hbar*self.OM_T[:,0])
        
        rk.plot_mat(amplit_tot_FT_mid, extent=[self.E_lo,self.E_hi,-self.em_axis_mid_reach,self.em_axis_mid_reach],
                saveloc='single_output_temp/pipeline_diag/raw_signal_FT_mid.png', show=False)
        
        amplit_tot_FT_mid_detrended, spike_only, spike_row_mid = rk.detrend_spike(amplit_tot_FT_mid, em_axis_mid, 0, 2, N_E=self.N_E, plot=False)
        
        rk.plot_mat(amplit_tot_FT_mid_detrended, extent=[self.E_lo,self.E_hi,-self.em_axis_mid_reach,self.em_axis_mid_reach],
                saveloc='single_output_temp/pipeline_diag/raw_signal_FT_mid_detrended.png', show=False)
        
        # Zero-pad the detrended mid-band to the full (ω, E) grid
        self.amplit_tot_FT_detrended_full = np.copy(self.amplit_tot_FT)
        self.amplit_tot_FT_detrended_full[i0:i1, :] = amplit_tot_FT_mid_detrended
        
        # Replace the omega-oscillation feature (positive & negative bands)
        # with uniform Rician bias from a feature-free donor band.
        #   mask_lo/mask_hi : indirect-energy band (eV) of the feature (positive side)
        #   donor_lo        : lower edge (eV) of a feature-free donor band of equal width
        om_t_eV = hbar * self.OM_T[:, 0]  # indirect energy axis in eV

        mask_lo_eV  = self.mask_lo           # e.g. 1.35
        mask_hi_eV  = self.mask_hi           # e.g. 1.75
        donor_lo_eV = self.donor_lo          # e.g. 0.60

        # --- positive band ---
        mask_i0_pos = np.argmin(np.abs(om_t_eV - mask_lo_eV))
        mask_i1_pos = np.argmin(np.abs(om_t_eV - mask_hi_eV))
        if mask_i0_pos > mask_i1_pos:
            mask_i0_pos, mask_i1_pos = mask_i1_pos, mask_i0_pos
        band_width = mask_i1_pos - mask_i0_pos

        donor_i0_pos = np.argmin(np.abs(om_t_eV - donor_lo_eV))
        donor_i1_pos = donor_i0_pos + band_width

        self.amplit_tot_FT_detrended_full[mask_i0_pos:mask_i1_pos, :] = \
            self.amplit_tot_FT[donor_i0_pos:donor_i1_pos, :]

        # --- negative band (mirror about 0 eV) ---
        mask_i0_neg = np.argmin(np.abs(om_t_eV - (-mask_hi_eV)))
        mask_i1_neg = np.argmin(np.abs(om_t_eV - (-mask_lo_eV)))
        if mask_i0_neg > mask_i1_neg:
            mask_i0_neg, mask_i1_neg = mask_i1_neg, mask_i0_neg

        donor_i0_neg = np.argmin(np.abs(om_t_eV - (-donor_lo_eV)))
        donor_i1_neg = donor_i0_neg - band_width
        if donor_i1_neg > donor_i0_neg:
            donor_i0_neg, donor_i1_neg = donor_i1_neg, donor_i0_neg

        self.amplit_tot_FT_detrended_full[mask_i0_neg:mask_i1_neg, :] = \
            self.amplit_tot_FT[donor_i0_neg:donor_i0_neg + band_width, :]

    
    def kb_correct(self):
        """Apply Koay-Basser correction for both full and probe signals."""
        # Koay-Basser full signal correction
        tot_rician_full = np.sum(self.signal_sb+self.b, axis=0)
        z_target = np.sum(self.signal_sb, axis=0)
        
        # Decompose RL reconstruction as a sum of n Gaussians and plot
        n_comp = 8
        # Use non-negative target for Gaussian fit
        tot_rician_pr_fit, _ = rk.fit_n_gaussians_1d(
            y_vals=self.om_probe,
            z_vals=z_target,
            n=n_comp
        )
        tot_rician_full_fit = tot_rician_pr_fit + self.b*self.N_T
        
        amp_corr = np.zeros_like(self.amplit_tot_FT_wndw)
        
        for j in range(self.N_E):
            col_now = np.abs(self.amplit_tot_FT_wndw[:,j])
            bias_now = tot_rician_full_fit[j]
            amp_corr_now = rk.koay_basser_correction(col_now, bias_now, lambda_thresh=0.8)
            amp_corr[:,j] = amp_corr_now
            if j%100==0:
                print(j)
        
        phase = np.angle(self.amplit_tot_FT_wndw)
        self.amplit_tot_FT_wndw = amp_corr * np.exp(1j*phase)
        
        rk.plot_mat(self.amplit_tot_FT_wndw+1e-20, extent=[self.E_lo,self.E_hi,hbar*self.OM_T[0,0],hbar*self.OM_T[-1,0]],
                saveloc='single_output_temp/pipeline_diag/signal_FT_Koay-Basser.png', show=False)
        
        # Koay-Basser probe signal correction
        # mixed signal time reach
        i_mix = floor(self.T_mix_reach/self.T_reach/2*self.N_T)
        
        # Remove rows [N_T//2 - i_mix : N_T//2 + i_mix] along time axis
        signal_xuv = np.delete(self.signal_sb, np.s_[self.N_T//2 - i_mix : self.N_T//2 + i_mix], axis=0)
        
        tot_rician = np.mean(signal_xuv, axis=0)*self.N_T*(2*self.T_reach/self.N_T)**2
        
        # Decompose RL reconstruction as a sum of n Gaussians and plot
        n_comp = 1
        # Use non-negative target for Gaussian fit
        tot_rician_fit, _ = rk.fit_n_gaussians_1d(
            y_vals=self.om_xuv,
            z_vals=tot_rician,
            n=n_comp
        )
        
        amp_corr = np.zeros_like(self.amplit_tot_FT_detrended_full)
        
        for j in range(self.N_E):
            col_now = np.abs(self.amplit_tot_FT_detrended_full[:,j])
            bias_now = tot_rician_fit[j]
            amp_corr_now = rk.koay_basser_correction(col_now, bias_now)
            amp_corr[:,j] = amp_corr_now
            if j%100==0:
                print(j)
        
        phase = np.angle(self.amplit_tot_FT_detrended_full)
        self.amplit_tot_FT_detrended_full = amp_corr * np.exp(1j*phase)
        
        rk.plot_mat(self.amplit_tot_FT_detrended_full, extent=[self.E_lo,self.E_hi,hbar*self.OM_T[0,0],hbar*self.OM_T[-1,0]],
                saveloc='single_output_temp/pipeline_diag/signal_FT_detrended_Koay-Basser.png', show=False)
        
        self.sig_probe_reconstructed,_,_,_ = rk.CFT(self.T_range, self.amplit_tot_FT_detrended_full, use_window=False, inverse=True)
        
        rk.plot_mat(self.sig_probe_reconstructed, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
                saveloc='single_output_temp/pipeline_diag/sig_probe_reconstructed.png', show=False)
        rk.plot_mat(self.synth_bsln, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
                saveloc='single_output_temp/pipeline_diag/sig_probe_ideal.png', show=False)
        rk.plot_mat((self.sig_probe_reconstructed - self.synth_bsln)/np.max(self.sig_probe_reconstructed),
                extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
                saveloc='single_output_temp/pipeline_diag/sig_probe_rec-id_difference.png', show=False)

    def xuv_peak(self):
        """Reconstruct XUV peak spectrum."""

        xuv_sigq_avg = np.sum(self.signal_harmq,axis=0)
        xuv_sigq_avg = np.sqrt(np.abs(xuv_sigq_avg))
        xuv_sigq_avg *= np.max(self.sp_xuv)/np.max(xuv_sigq_avg)

        # Decompose RL reconstruction as a sum of n Gaussians and plot
        n_comp = 2
        # Use non-negative target for Gaussian fit
        self.sp_xuv_meas_sig_fit, sp_xuv_meas_sig_fit_params = rk.fit_n_gaussians_1d(
            y_vals=self.E[0,:]/hbar,
            z_vals=xuv_sigq_avg,
            n=n_comp
        )
        
        self.xuvs_rec = rk.nfit_params_to_probes(sp_xuv_meas_sig_fit_params, self.T)
        self.sp_xuv_meas_sig_fit = rk.sp_tot(self.xuvs_rec,self.om_xuv)

        self.xuvs_rec = self.xuvs
        self.sp_xuv_meas_sig_fit = self.sp_xuv
        
        # Convert energy (ħ·ω in eV) to wavelength in nm and sort ascending
        E_eV = self.om_xuv * hbar
        lambda_nm = 1239.84197386209 / E_eV
        idx = np.argsort(lambda_nm)

        print(self.xuvs_rec[0][2]*hbar - self.xuvs_rec[1][2]*hbar)
        
        # plt.plot(lambda_nm[idx],self.sp_xuv[idx],label='true xuv spectrum')
        # plt.plot(lambda_nm[idx],self.sp_xuv_meas_sig_fit[idx],linewidth=0.65,linestyle='--',label='reconstructed xuv sp.')
        
        plt.plot(self.E[0,:],xuv_sigq_avg,label='true xuv spectrum')
        plt.plot(self.om_xuv*hbar,self.sp_xuv_meas_sig_fit,linewidth=0.65,linestyle='--',label='reconstructed xuv sp.')
        plt.xlabel('lambda [nm]')
        plt.ylabel('Amplitude (normalized)')
        plt.title('XUV spectrum rec, single gaussian fit')
        plt.legend()
        plt.tight_layout()
        plt.savefig('single_output_temp/reconstructions/sp_xuv_rec.png',dpi=300)
        plt.close()
    
    def WF_reconstruct(self):
        """Retrieve probe spectrum using Wirtinger Flow."""
        
        sig_probe_reconstructed = self.sig_probe_reconstructed + self.b
        
        sp_rec, lasterr = rk.reconstruct_WirtFlow(sig_probe_reconstructed, self.sp_probe, 
                                             self.sp_xuv_meas_sig_fit, self.om_probe, self.om_xuv, 
                                             self.T, self.b, n_power_iter=50,
                                             n_main_iter=3000, ifplot=50, median_regval=4, 
                                             lastmax_margin=np.sqrt(self.alpha)*700*np.sqrt(self.N_T/350), eps=3e-9,
                                             ifwait=False, alph=self.alpha, nt=self.N_T)
        
        # Decompose RL reconstruction as a sum of n Gaussians and plot
        n_comp = 8
        om_grid = self.om_probe
        # Use non-negative target for Gaussian fit
        
        z_target = np.abs(sp_rec)
        
        fit_gauss, fit_params = rk.fit_n_gaussians_1d(
            y_vals=om_grid,
            z_vals=z_target,
            n=n_comp
        )

        self.probes_reconstructed = rk.nfit_params_to_probes(fit_params, self.T)
        
        plt.figure()
        # Convert energy (ħ·ω in eV) to wavelength in nm and sort ascending

        # E_eV = om_grid * hbar
        # lambda_nm = 1239.84197386209 / E_eV
        # idx = np.argsort(lambda_nm)[:-30]
        
        # plt.plot(lambda_nm[idx], rk.normalize_abs(z_target)[idx], label='WF target')
        # plt.plot(lambda_nm[idx], rk.normalize_abs(fit_gauss)[idx], label=f'{n_comp}-Gaussian fit')
        # plt.plot(lambda_nm[idx], rk.normalize_abs(self.sp_probe)[idx], label='True spectrum')

        plt.plot(om_grid, rk.normalize_abs(z_target), label='WF target')
        plt.plot(om_grid, rk.normalize_abs(fit_gauss), label=f'{n_comp}-Gaussian fit')
        plt.plot(om_grid, rk.normalize_abs(self.sp_probe), label='True spectrum')

        plt.xlabel('lambda [nm] ACTUALLY NO, IT NEEDS FIX')
        plt.ylabel('Amplitude (normalized)')
        plt.title('n-Gaussian decomposition of RL reconstruction')
        plt.legend()
        plt.tight_layout()
        plt.savefig('single_output_temp/reconstructions/final_sp_probe_rec.png',dpi=300)
        plt.close()
        
        # === Analytic Gaussian correction (correcting_function_multi) ===
        correction = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs, self.om_xuv), 
                                             rk.normalize_params(self.probes, self.om_probe), 
                                             dzeta=self.dzeta_val, theta=self.theta_val)
        self.amplit_tot_FT_corrected = correction*self.amplit_tot_FT_wndw
        self.amplit_tot_FT_corrected = median_filter(np.abs(self.amplit_tot_FT_corrected), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected))
        
        correction_x = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs_rec, self.om_xuv), 
                                             rk.normalize_params(self.probes, self.om_probe), 
                                             dzeta=self.dzeta_val, theta=self.theta_val)
        self.amplit_tot_FT_corrected_x = correction_x*self.amplit_tot_FT_wndw
        self.amplit_tot_FT_corrected_x = median_filter(np.abs(self.amplit_tot_FT_corrected_x), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected_x))
        
        correction_rec_x = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs_rec, self.om_xuv), 
                                             rk.normalize_params(self.probes_reconstructed, self.om_probe), 
                                             dzeta=self.dzeta_val, theta=self.theta_val)
        self.amplit_tot_FT_corrected_rec_x_nomedian = correction_rec_x*self.amplit_tot_FT_wndw
        self.amplit_tot_FT_corrected_rec_x = median_filter(np.abs(self.amplit_tot_FT_corrected_rec_x_nomedian), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected_rec_x_nomedian))

        # # === Synthetic signal ratio correction (correcting_function_synth) ===
        # sp_ref_corr = rk.sp_tot(self.refs, self.om_probe_up)

        # # --- Correction 1: true XUV + true probe ---
        # sp_xuv_corr = rk.sp_tot(self.xuvs, self.om_xuv_up)
        # sp_probe_corr = rk.sp_tot(self.probes, self.om_probe_up)
        # correction = rk.correcting_function_synth(
        #     self.T_range, self.T_up,
        #     sp_probe_corr, sp_xuv_corr, self.om_probe_up, self.om_xuv_up,
        #     sp_ref=sp_ref_corr, p_E=self.p_E,
        #     dzeta=self.dzeta_val, theta=self.theta_val)
        # self.amplit_tot_FT_corrected = correction * self.amplit_tot_FT_wndw
        # self.amplit_tot_FT_corrected = median_filter(np.abs(self.amplit_tot_FT_corrected), size=(3,3)) * np.exp(1j*np.angle(self.amplit_tot_FT_corrected))

        # # --- Correction 2: reconstructed XUV + true probe ---
        # sp_xuv_rec = rk.sp_tot(self.xuvs_rec, self.om_xuv_up)
        # correction_x = rk.correcting_function_synth(
        #     self.T_range, self.T_up,
        #     sp_probe_corr, sp_xuv_rec, self.om_probe_up, self.om_xuv_up,
        #     sp_ref=sp_ref_corr, p_E=self.p_E,
        #     dzeta=self.dzeta_val, theta=self.theta_val)
        # self.amplit_tot_FT_corrected_x = correction_x * self.amplit_tot_FT_wndw
        # self.amplit_tot_FT_corrected_x = median_filter(np.abs(self.amplit_tot_FT_corrected_x), size=(3,3)) * np.exp(1j*np.angle(self.amplit_tot_FT_corrected_x))

        # # --- Correction 3: reconstructed XUV + reconstructed probe ---
        # sp_probe_rec = rk.sp_tot(self.probes_reconstructed, self.om_probe_up)
        # correction_rec_x = rk.correcting_function_synth(
        #     self.T_range, self.T_up,
        #     sp_probe_rec, sp_xuv_rec, self.om_probe_up, self.om_xuv_up,
        #     sp_ref=sp_ref_corr, p_E=self.p_E,
        #     dzeta=self.dzeta_val, theta=self.theta_val)
        # self.amplit_tot_FT_corrected_rec_x_nomedian = correction_rec_x * self.amplit_tot_FT_wndw
        # self.amplit_tot_FT_corrected_rec_x = median_filter(np.abs(self.amplit_tot_FT_corrected_rec_x_nomedian), size=(3,3)) * np.exp(1j*np.angle(self.amplit_tot_FT_corrected_rec_x_nomedian))
    
    def resample_analyze(self):
        """Resample and analyze results to produce final density matrices."""
        # Resample and analyze

        rho_uncorrected, amplit_tot_FT_corrected_small, extent_small, idxs_small, _, _ = rk.resample(
            median_filter(np.abs(self.amplit_tot_FT_wndw),size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_wndw)), self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
        
        rk.plot_mat(rho_uncorrected, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/synth_uncorr_unproj.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho uncorrected for the probe spectrum', show=False)
        
        rho_uncorrected = rk.project_to_density_matrix(rho_uncorrected)

        rho_reconstructed, amplit_tot_FT_corrected_small, extent_small, idxs_small, E1, E2 = rk.resample(
            self.amplit_tot_FT_corrected, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
        rho_reconstructed = rk.project_to_density_matrix(rho_reconstructed)
        
        rho_reconstructed_x, amplit_tot_FT_corrected_small, extent_small, idxs_small, _, _ = rk.resample(
            self.amplit_tot_FT_corrected_x, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
        rho_reconstructed_x = rk.project_to_density_matrix(rho_reconstructed_x)
        
        rho_reconstructed_rec_x, amplit_tot_FT_corrected_small_rec, extent_small, idxs_small, _, _ = rk.resample(
            self.amplit_tot_FT_corrected_rec_x, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
        rho_reconstructed_rec_x = rk.project_to_density_matrix(rho_reconstructed_rec_x)
        
        rho_reconstructed_rec_x_nomedian, amplit_tot_FT_corrected_small_rec, extent_small, idxs_small, _, _ = rk.resample(
            self.amplit_tot_FT_corrected_rec_x_nomedian, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
        rho_reconstructed_rec_x_nomedian = rk.project_to_density_matrix(rho_reconstructed_rec_x_nomedian)
        
        # Comparison with a ground truth

        plt.plot(E1[:,0],rk.sp_tot(self.xuvs, E1[:,0]/hbar))
        plt.savefig('fig1.png')
        plt.close()

        if self.if_coherent:
            ### Ground truth - coherent sum
            xuv_gt = rk.sp_tot(self.xuvs, E1[:,0]/hbar)
            ideal_rho = xuv_gt[:,np.newaxis]@np.conjugate(xuv_gt[np.newaxis,:])
            ideal_rho = rk.project_to_density_matrix(ideal_rho, smooth_sigma=0.1)
        else:
            ### Ground truth - incoherent sum of individual XUV component density matrices
            ideal_rho = np.zeros((len(E1[:,0]), len(E1[:,0])), dtype=complex)
            for xuv_i in self.xuvs:
                xuv_gt_i = rk.sp_tot((xuv_i,), E1[:,0]/hbar)
                ideal_rho += xuv_gt_i[:,np.newaxis]@np.conjugate(xuv_gt_i[np.newaxis,:])
            ideal_rho = rk.project_to_density_matrix(ideal_rho, smooth_sigma=0.1)

        ### Ground truth - arbitrary shape
        # ideal_rho =  np.exp(-((E1 - self.om0_xuv*hbar)/self.s_xuv)**2 - ((E2 - self.om0_xuv*hbar)/self.s_xuv)**2)
        # ideal_rho = rk.project_to_density_matrix(ideal_rho)

        fid0 = rk.fidelity(ideal_rho, rho_uncorrected)
        fid1 = rk.fidelity(ideal_rho, rho_reconstructed)
        fid2 = rk.fidelity(ideal_rho, rho_reconstructed_x)
        fid3 = rk.fidelity(ideal_rho, rho_reconstructed_rec_x)
        fid4 = rk.fidelity(ideal_rho, rho_reconstructed_rec_x_nomedian)
        
        print(fid0,fid1,fid2,fid3,fid4)
        
        rk.plot_mat(ideal_rho, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/ideal_rho.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='True quantum state', show=False, caption='F=1.00')
        
        rk.plot_mat(rho_uncorrected, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/synth_uncorr.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho uncorrected for the probe spectrum', show=False, caption='F=%.3f'%fid0)
        
        rk.plot_mat(rho_reconstructed, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/synth_corr.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho ideally corrected for the probe spectrum', show=False, caption='F=%.3f'%fid1)
        
        rk.plot_mat(rho_reconstructed_x, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/synth_corr_x.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho rec xuv and ideally corrected for the probe spectrum', show=False, caption='F=%.3f'%fid2)
        
        rk.plot_mat(rho_reconstructed_rec_x, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='phase', saveloc='single_output_temp/rhos/rec_corr_x.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho rec xuv and WF rec probe', show=False, caption='F=%.3f'%fid3)
        
        rk.plot_mat(rho_reconstructed_rec_x_nomedian, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/rec_corr_x_nomedian.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho rec xuv and WF rec probe', show=False, caption='F=%.3f'%fid4)

        # Store results as instance attributes
        self.rho_uncorrected = rho_uncorrected
        self.rho_reconstructed = rho_reconstructed
        self.rho_reconstructed_x = rho_reconstructed_x
        self.rho_reconstructed_rec_x = rho_reconstructed_rec_x
        self.rho_reconstructed_rec_x_nomedian = rho_reconstructed_rec_x_nomedian
        self.ideal_rho = ideal_rho
        self.fidelities = [fid0, fid1, fid2, fid3, fid4]


if __name__ == "__main__":

    N_T_range = np.linspace(200,1000,10)
    # alpha_range = 10000*400/N_T_range
    alpha_range = np.linspace(10000*4/10,10000*4/2,10)

    # Create meshgrids for 3D surface plots
    ALPHA, N_T_ar = np.meshgrid(alpha_range, N_T_range)


    fid1s = np.zeros_like(ALPHA)
    fid2s = np.zeros_like(ALPHA)
    fid3s = np.zeros_like(ALPHA)
    fid4s = np.zeros_like(ALPHA)
    PSBC = np.zeros_like(ALPHA)

    for i_nt in range(len(N_T_range)):
        for i_alpha in range(len(alpha_range)):

            alpha = ALPHA[i_alpha,i_nt]
            N_T = int(N_T_ar[i_alpha,i_nt])
            print(i_alpha,i_nt)

            # Example usage of the RK_experiment class

            E_lo = 23.0
            # E_hi = 29
            E_hi = 26.7
            T_reach = 50
            # T_reach = 250
            E_res = 0.025    
            # E_res = 0.005    
            N_T = 260
            # N_T = 1430
            p_E = 4  # N_E upsampling integer
            alpha = 10000
            # alpha = 10000
            b = 1

            sideband_lo = 24.7
            sideband_hi = 26.6
            harmq_lo = 23.3

            if_coherent = False

            # Create experiment instance
            experiment = RK_experiment(E_lo=E_lo,E_hi=E_hi,T_reach=T_reach,E_res=E_res,N_T=N_T,p_E=p_E,alpha=alpha,b=b,
                                    sb_lo=sideband_lo,sb_hi=sideband_hi,harmq_lo=harmq_lo,if_coherent=if_coherent)
            
            # Define pulses
            # A_xuv = 1.0
            # a_xuvs = [1.0,1.0]
            # om_xuvs = [10.65/hbar,(10.65+2*1.55)/hbar]
            # s_xuvs = [0.15/hbar,0.18/hbar]

            # A_xuv = 0.1
            # a_xuvs = [1.0]
            # om_xuvs = [(25.65-1*1.40)/hbar]
            # s_xuvs = [0.15/hbar]

            A_xuv = 0.1
            a_xuvs = [np.sqrt(0.4),np.sqrt(0.8)]
            om_xuvs = [(25.65-1*1.75)/hbar,(25.65-1*1.75+0.27)/hbar]
            s_xuvs = [0.10/hbar,0.10/hbar]

            A_probe = 1.2
            a_probes = [1.0,0.2,0.2,0.3]
            om_probes = [1.55/hbar,1.20/hbar,2.00/hbar,1.85/hbar]
            s_probes = [0.15/hbar,0.04/hbar,0.07/hbar,0.17/hbar]

            # # Flat-top probe (similar total width and avg amplitude)
            # A_probe = 1.2
            # a_probes = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35]
            # om_probes = [1.25/hbar, 1.39/hbar, 1.53/hbar, 1.67/hbar, 1.81/hbar, 1.95/hbar, 2.09/hbar]
            # s_probes = [0.10/hbar, 0.10/hbar, 0.10/hbar, 0.10/hbar, 0.10/hbar, 0.10/hbar, 0.10/hbar]

            A_ref = 0.6
            a_refs = [1.0,0.01]
            om_refs = [1.55/hbar,1.55/hbar]
            s_refs = [0.07/hbar,0.15/hbar]

            experiment.define_pulses(A_xuv,A_probe,A_ref,a_xuvs,a_probes,a_refs,om_xuvs,om_probes,om_refs,s_xuvs,s_probes,s_refs)
            
            # Run the full pipeline
            experiment.generate_signal()
            PSBC[i_alpha,i_nt] = experiment.peak_sb_counts

            try:

                experiment.process_and_detrend()
                experiment.kb_correct()
                # experiment.xuv_peak()
                experiment.WF_reconstruct()
                experiment.resample_analyze()

                fid1s[i_alpha,i_nt] = experiment.fidelities[1]
                fid2s[i_alpha,i_nt] = experiment.fidelities[2]
                fid3s[i_alpha,i_nt] = experiment.fidelities[3]
                fid4s[i_alpha,i_nt] = experiment.fidelities[4]

            except:
                fid1s[i_alpha,i_nt] = 0
                fid2s[i_alpha,i_nt] = 0
                fid3s[i_alpha,i_nt] = 0
                fid4s[i_alpha,i_nt] = 0

            exit()
            
            # np.save('scans/newscan/fid1s400incoh.npy',fid1s)
            # np.save('scans/newscan/fid2s400incoh.npy',fid2s)
            # np.save('scans/newscan/fid3s400incoh.npy',fid3s)
            # np.save('scans/newscan/fid4s400incoh.npy',fid4s)
            # np.save('scans/newscan/PSBCincoh.npy',PSBC)