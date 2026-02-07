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
    
    def __init__(self,E_lo=60.0,E_hi=63.5,T_reach=50,E_res=0.025,N_T=240,p_E=4,alpha=0.05,b=1,sb_lo=24.7,sb_hi=28,harmq_lo=22):
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
        
        # Noise and processing parameters
        self.ifnoise = True
        # self.noise_area_Elo = 0.0  NOISE FLOOR SHOULD BE QUITE PRECISELY MEASURED BY CAPTURING WITH LASER OFF
        # self.noise_area_Ehi = 0.1
        self.sb_lo = sb_lo
        self.sb_hi = sb_hi
        self.harmq_lo = harmq_lo
        
        # Detrending parameters
        self.em_axis_mid_reach = 1.00
        self.T_mix_reach = 40
        self.sidebands_reach = 2.15
        
        # Correction parameters
        self.dzeta_val = 1e-3
        self.theta_val = 0.01
        
        # Resampling parameters
        self.rho_lo = harmq_lo
        self.rho_hi = sb_lo
        
        # Random number generator
        self.rng = np.random.default_rng()
        
        # Initialize grids and pulses
        self.prepare_grid()
    
    def prepare_grid(self):
        """Initialize energy and time grids and pulse configurations."""
        self.E_range = np.linspace(self.E_lo, self.E_hi, self.N_E)
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
    
    def generate_signal(self):
        """Generate synthetic baseline and full signal with optional noise."""
        # Synthetic baseline with known spectra
        self.om_probe = (self.E/hbar - ((self.E_lo/hbar+self.E_hi/hbar)/2-self.om_ref) + self.E_span/hbar/self.N_E/2 * ((self.N_E-1) % 2))[0,:]
        self.om_xuv = (self.E/hbar - self.om_ref)[0,:]
        
        self.sp_xuv = rk.sp_tot(self.xuvs, self.om_xuv)
        self.sp_probe = rk.sp_tot(self.probes, self.om_probe)
        self.sp_ref = rk.sp_tot(self.refs, self.om_probe)
        
        self.om_probe_up = (self.E_up/hbar - ((self.E_lo/hbar+self.E_hi/hbar)/2-self.om_ref) + self.E_span/hbar/self.N_E_up/2 * ((self.N_E_up-1) % 2))[0,:]
        self.om_xuv_up = (self.E_up/hbar - self.om_ref)[0,:]

        self.sp_xuv_up = rk.sp_tot(self.xuvs, self.om_xuv_up)
        self.sp_probe_up = rk.sp_tot(self.probes, self.om_probe_up)
        self.sp_ref_up = rk.sp_tot(self.refs, self.om_probe_up)
        
        om_probe_up_emit = (self.E_up/hbar - ((self.E_lo/hbar+self.E_hi/hbar)/2+self.om_ref) + self.E_span/hbar/self.N_E_up/2 * ((self.N_E_up-1) % 2))[0,:]
        om_xuv_up_emit = (self.E_up/hbar + self.om_ref)[0,:]
        
        sp_xuv_up_emit = rk.sp_tot(self.xuvs, om_xuv_up_emit)
        sp_probe_up_emit = rk.sp_tot(rk.pulse_emit(self.probes), om_probe_up_emit)
        sp_ref_up_emit = rk.sp_tot(rk.pulse_emit(self.refs), om_probe_up_emit)

        test_conv = fftconvolve(self.sp_xuv_up,self.sp_probe_up,mode="same")

        # Plot the upsampled spectra with probe/ref on one axis, xuv on another
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        
        # Top subplot: Probe and Reference spectra
        ax1.plot(om_probe_up_emit * hbar, sp_probe_up_emit, label='Probe spectrum', linewidth=2)
        ax1.plot(om_probe_up_emit * hbar, sp_ref_up_emit, label='Reference spectrum', linewidth=2)
        ax1.set_xlabel('Energy [eV]')
        ax1.set_ylabel('Amplitude [arb. u.]')
        ax1.set_title('Probe and Reference Spectra')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot: XUV spectrum
        ax2.plot(self.om_xuv_up * hbar, self.sp_xuv_up, label='XUV spectrum', linewidth=2, color='orange')
        ax2.set_xlabel('Energy [eV]')
        ax2.set_ylabel('Amplitude [arb. u.]')
        ax2.set_title('XUV Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(test_conv)
        
        plt.tight_layout()
        plt.savefig('single_output_temp/spectra/input_spectra.png', dpi=300)
        plt.close()
        
        # Probe only for reference
        self.synth_bsln = np.abs(rk.downsample(rk.synth_baseline(self.sp_probe_up, self.sp_xuv_up, 
                                                          rk.regularize_omega(self.om_probe_up), rk.regularize_omega(self.om_xuv_up), self.T_up), self.p_E))**2
        self.synth_bsln = self.rng.poisson(self.alpha*(self.synth_bsln)+self.b).astype(float) - self.b
        self.synth_bsln_FT, _, el, eh = rk.CFT(self.T_range, self.synth_bsln, use_window=False)
        
        # Synthetic full signal
        sp_xuv_E = rk.sp_tot(self.xuvs,self.E_up/hbar)
        amplit_tot_0 = rk.downsample(
                                rk.synth_baseline(self.sp_probe_up, self.sp_xuv_up, 
                                            rk.regularize_omega(self.om_probe_up), rk.regularize_omega(self.om_xuv_up), self.T_up)
                                + rk.synth_baseline(self.sp_ref_up, self.sp_xuv_up, 
                                            rk.regularize_omega(self.om_probe_up), rk.regularize_omega(self.om_xuv_up), self.T_up*0)
                                # + rk.synth_baseline(sp_probe_up_emit, sp_xuv_up_emit, 
                                #             rk.regularize_omega(om_probe_up_emit), rk.regularize_omega(om_xuv_up_emit), self.T_up) 
                                # + rk.synth_baseline(sp_ref_up_emit, sp_xuv_up_emit, 
                                #             rk.regularize_omega(om_probe_up_emit), rk.regularize_omega(om_xuv_up_emit), self.T_up*0)
                                + sp_xuv_E
                                ,self.p_E)
        signal_clean = np.abs(amplit_tot_0)**2
        signal_clean *= self.alpha
        signal_clean += self.b
        
        if not self.ifnoise:
            self.signal = signal_clean.copy()
        else:
            # Apply Poisson noise assuming signal_clean contains expected values (means)
            self.signal = self.rng.poisson(signal_clean).astype(float)
        
        print("N Total = %.2e"%np.sum(self.signal))
    
    def process_and_detrend(self):
        """Process signal, estimate background, and perform detrending."""
        # Processing starts here
        # self.b_est = np.mean(self.signal[:,floor(self.noise_area_Elo*self.N_E):floor(self.noise_area_Ehi*self.N_E)])
        # self.signal -= self.b_est

        self.signal -= self.b # NOISE FLOOR SHOULD BE QUITE PRECISELY MEASURED BY CAPTURING WITH LASER OFF

        # Zero out signal outside the sideband range [sb_lo, sb_hi]
        sb_lo_idx = np.argmin(np.abs(self.E_range - self.sb_lo))
        sb_hi_idx = np.argmin(np.abs(self.E_range - self.sb_hi))
        self.signal_sb = np.zeros_like(self.signal)
        self.signal_sb[:, sb_lo_idx:sb_hi_idx+1] = self.signal[:, sb_lo_idx:sb_hi_idx+1]

        harmq_lo_idx = np.argmin(np.abs(self.E_range - self.harmq_lo))
        self.signal_harmq = np.zeros_like(self.signal)
        self.signal_harmq[:, harmq_lo_idx:sb_lo_idx+1] = self.signal[:, harmq_lo_idx:sb_lo_idx+1]

        # self.signal = self.signal_sb
        
        peak_row_counts = np.max(np.sum(self.signal, axis=1))
        print(peak_row_counts/self.alpha)
        
        rk.plot_mat(self.signal, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
                caption='peak row counts: %.2e\nN_T: %i\nN_E: %i\nT_res: %.2f fs\nE_res: %.3f eV'%(
                    peak_row_counts,self.N_T,self.N_E,2*self.T_reach/self.N_T,self.E_res),
                saveloc='single_output_temp/pipeline_diag/measured_signal.png')
        
        rk.plot_mat(self.signal_sb, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
                caption='peak row counts: %.2e\nN_T: %i\nN_E: %i\nT_res: %.2f fs\nE_res: %.3f eV'%(
                    peak_row_counts,self.N_T,self.N_E,2*self.T_reach/self.N_T,self.E_res),
                saveloc='single_output_temp/pipeline_diag/measured_signal_sb.png')
        
        rk.plot_mat(self.signal_harmq, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
                caption='peak row counts: %.2e\nN_T: %i\nN_E: %i\nT_res: %.2f fs\nE_res: %.3f eV'%(
                    peak_row_counts,self.N_T,self.N_E,2*self.T_reach/self.N_T,self.E_res),
                saveloc='single_output_temp/pipeline_diag/measured_signal_harmq.png')
        
        self.amplit_tot_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal_sb, use_window=False)
        self.amplit_tot_FT_wndw, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal_sb, use_window=True)
        
        rk.plot_mat(self.amplit_tot_FT_wndw, extent=[self.E_lo,self.E_hi,em_lo,em_hi],
                saveloc='single_output_temp/pipeline_diag/raw_signal_FT.png', show=False)
        
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
        
        # Extent the poissonian noise with rician bias
        sideband_lo, sideband_hi = floor(0.5*(1 - self.sidebands_reach/em_hi)*self.N_T), floor(0.5*(1 + self.sidebands_reach/em_hi)*self.N_T)
        self.amplit_tot_FT_detrended_full[sideband_lo:i0,:] = self.amplit_tot_FT[0:i0-sideband_lo,:]
        self.amplit_tot_FT_detrended_full[i1:sideband_hi,:] = self.amplit_tot_FT[self.N_T-sideband_hi+i1:,:]

    
    def kb_correct(self):
        """Apply Koay-Basser correction for both full and probe signals."""
        # Koay-Basser full signal correction
        tot_rician_full = np.sum(self.signal_sb+self.b, axis=0)
        z_target = np.sum(self.signal_sb, axis=0)
        
        # Decompose RL reconstruction as a sum of n Gaussians and plot
        n_comp = 12
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
            amp_corr_now = rk.koay_basser_correction(col_now, bias_now, lambda_thresh=1)
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
        xuv_spq_avg = np.sqrt(np.abs(xuv_sigq_avg))
        
        # Decompose RL reconstruction as a sum of n Gaussians and plot
        n_comp = 2
        xuv_spq_fit, _ = rk.fit_n_gaussians_1d(
            y_vals=self.E[0,:],
            z_vals=xuv_spq_avg,
            n=n_comp
        )
        
        xuv_spq_fit *= np.max(self.sp_xuv)/np.max(xuv_spq_fit)
        
        # Align fit_gauss's maximum (along energy) with sp_xuv's maximum
        idx_xuv = int(np.argmax(self.sp_xuv))
        idx_spk = int(np.argmax(xuv_spq_fit))
        shift_cols = idx_xuv - idx_spk
        xuv_spq_fit = np.roll(xuv_spq_fit, shift_cols)
        
        # Decompose RL reconstruction as a sum of n Gaussians and plot
        n_comp = 1
        # Use non-negative target for Gaussian fit
        self.sp_xuv_meas_sig_fit, sp_xuv_meas_sig_fit_params = rk.fit_n_gaussians_1d(
            y_vals=self.om_xuv,
            z_vals=xuv_spq_fit,
            n=n_comp
        )
        
        self.xuvs_rec = rk.nfit_params_to_probes(sp_xuv_meas_sig_fit_params, self.T)
        
        # Convert energy (ħ·ω in eV) to wavelength in nm and sort ascending
        E_eV = self.om_xuv * hbar
        lambda_nm = 1239.84197386209 / E_eV
        idx = np.argsort(lambda_nm)
        
        plt.plot(lambda_nm[idx],self.sp_xuv[idx],label='true xuv spectrum')
        plt.plot(lambda_nm[idx],self.sp_xuv_meas_sig_fit[idx],linewidth=0.65,linestyle='--',label='reconstructed xuv sp.')
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
                                             lastmax_margin=np.sqrt(self.alpha)*700*np.sqrt(self.N_T/350),
                                             ifwait=False, alph=self.alpha, nt=self.N_T)
        
        # Decompose RL reconstruction as a sum of n Gaussians and plot
        n_comp = 8
        om_grid = self.om_probe
        # Use non-negative target for Gaussian fit
        
        z_target = np.abs(sp_rec)

        plt.plot(om_grid,z_target)
        plt.savefig('./sb.png')
        plt.close()
        
        fit_gauss, fit_params = rk.fit_n_gaussians_1d(
            y_vals=om_grid,
            z_vals=z_target,
            n=n_comp
        )

        plt.plot(om_grid,z_target)
        plt.plot(om_grid,fit_gauss)
        plt.savefig('./sb.png')
        plt.close()
        
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
        
        # Apply corrections using various probe configurations
        correction = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs, self.om_xuv)[0], 
                                             rk.normalize_params(self.probes, self.om_probe), 
                                             dzeta=self.dzeta_val, theta=self.theta_val)
        self.amplit_tot_FT_corrected = correction*self.amplit_tot_FT_wndw
        self.amplit_tot_FT_corrected = median_filter(np.abs(self.amplit_tot_FT_corrected), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected))
        
        correction_x = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs_rec, self.E[0,:])[0], 
                                               rk.normalize_params(self.probes, self.om_probe), 
                                               dzeta=self.dzeta_val, theta=self.theta_val)
        self.amplit_tot_FT_corrected_x = correction_x*self.amplit_tot_FT_wndw
        self.amplit_tot_FT_corrected_x = median_filter(np.abs(self.amplit_tot_FT_corrected_x), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected_x))
        
        correction_rec_x = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs_rec, self.E[0,:])[0], 
                                                   rk.normalize_params(self.probes_reconstructed, self.om_probe), 
                                                   dzeta=self.dzeta_val, theta=self.theta_val)
        self.amplit_tot_FT_corrected_rec_x_nomedian = correction_rec_x*self.amplit_tot_FT_wndw
        self.amplit_tot_FT_corrected_rec_x = median_filter(np.abs(self.amplit_tot_FT_corrected_rec_x_nomedian), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected_rec_x_nomedian))


    
    def resample_analyze(self):
        """Resample and analyze results to produce final density matrices."""
        # Resample and analyze
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
        ideal_rho = np.exp(-((E1 - self.om0_xuv*hbar)/self.s_xuv)**2 - ((E2 - self.om0_xuv*hbar)/self.s_xuv)**2)
        ideal_rho = rk.project_to_density_matrix(ideal_rho)
        
        fid1 = rk.fidelity(ideal_rho, rho_reconstructed)
        fid2 = rk.fidelity(ideal_rho, rho_reconstructed_x)
        fid3 = rk.fidelity(ideal_rho, rho_reconstructed_rec_x)
        fid4 = rk.fidelity(ideal_rho, rho_reconstructed_rec_x_nomedian)
        
        print(fid1,fid2,fid3,fid4)
        
        rk.plot_mat(rho_reconstructed, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/synth_corr.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho ideally corrected for the probe spectrum', show=False, caption='F=%.3f'%fid1)
        
        rk.plot_mat(rho_reconstructed_x, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/synth_corr_x.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho rec xuv and ideally corrected for the probe spectrum', show=False, caption='F=%.3f'%fid2)
        
        rk.plot_mat(rho_reconstructed_rec_x, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/rec_corr_x.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho rec xuv and WF rec probe', show=False, caption='F=%.3f'%fid3)
        
        rk.plot_mat(rho_reconstructed_rec_x_nomedian, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/rec_corr_x_nomedian.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho rec xuv and WF rec probe', show=False, caption='F=%.3f'%fid4)
        
        # Store results as instance attributes
        self.rho_reconstructed = rho_reconstructed
        self.rho_reconstructed_x = rho_reconstructed_x
        self.rho_reconstructed_rec_x = rho_reconstructed_rec_x
        self.rho_reconstructed_rec_x_nomedian = rho_reconstructed_rec_x_nomedian
        self.ideal_rho = ideal_rho
        self.fidelities = [fid1, fid2, fid3, fid4]


# Example usage of the RK_experiment class
if __name__ == "__main__":

    E_lo = 23.0
    E_hi = 29
    T_reach = 50
    E_res = 0.025    
    N_T = 240
    p_E = 4  # N_E upsampling integer
    alpha = 10000
    b = 1

    sideband_lo = 24.7
    sideband_hi = 26.6
    harmq_lo = 23.3

    # Create experiment instance
    experiment = RK_experiment(E_lo=E_lo,E_hi=E_hi,T_reach=T_reach,E_res=E_res,N_T=N_T,p_E=p_E,alpha=alpha,b=b,sb_lo=sideband_lo,sb_hi=sideband_hi,harmq_lo=harmq_lo)
    
    # Define pulses
    # A_xuv = 1.0
    # a_xuvs = [1.0,1.0]
    # om_xuvs = [10.65/hbar,(10.65+2*1.55)/hbar]
    # s_xuvs = [0.15/hbar,0.18/hbar]
    A_xuv = 0.1
    a_xuvs = [1.0,0.2,0.2]
    om_xuvs = [(25.65-1*1.55)/hbar,(25.65+1*1.50)/hbar,(25.65+1*1.65)/hbar]
    s_xuvs = [0.15/hbar,0.10/hbar,0.10/hbar]

    A_probe = 1.2
    a_probes = [1.0,0.2,0.2,0.3]
    om_probes = [1.55/hbar,1.20/hbar,2.00/hbar,1.85/hbar]
    s_probes = [0.15/hbar,0.04/hbar,0.07/hbar,0.17/hbar]

    A_ref = 0.6
    a_refs = [1.0,0.1]
    om_refs = [1.55/hbar,1.55/hbar]
    s_refs = [0.07/hbar,0.15/hbar]

    experiment.define_pulses(A_xuv,A_probe,A_ref,a_xuvs,a_probes,a_refs,om_xuvs,om_probes,om_refs,s_xuvs,s_probes,s_refs)
    
    # Run the full pipeline
    experiment.generate_signal()
    experiment.process_and_detrend()
    experiment.kb_correct()
    experiment.xuv_peak()
    experiment.WF_reconstruct()
    experiment.resample_analyze()