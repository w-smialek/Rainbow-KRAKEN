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
import rkraken1 as rk

# Reduced Planck constant in eV*fs (approx CODATA): ħ = 6.582119569e-16 eV·s
hbar = 6.582119569e-1

def plot_spectra(om_pr,om_x,sp_pr,sp_ref,sp_x):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Top subplot: Probe and Reference spectra
    ax1.plot(om_pr * hbar, sp_pr, label='Probe spectrum', linewidth=2)
    ax1.plot(om_pr * hbar, sp_ref, label='Reference spectrum', linewidth=2)
    ax1.set_xlabel('Energy [eV]',fontweight='bold')
    ax1.set_ylabel('Amplitude [arb. u.]',fontweight='bold')
    ax1.set_title('Probe and Reference Spectra',fontweight='bold',fontsize=12)
    ax1.set_xlim([1,2])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: XUV spectrum
    ax2.plot(om_x * hbar, sp_x, label='XUV spectrum', linewidth=2, color='purple')
    ax2.set_xlabel('Energy [eV]',fontweight='bold')
    ax2.set_ylabel('Amplitude [arb. u.]',fontweight='bold')
    ax2.set_title('XUV Spectrum',fontweight='bold',fontsize=12)
    ax2.set_xlim([24,26])
    # ax2.legend()
    ax2.grid(True, alpha=0.3)
    
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

    def define_pulses(self,A_xuv,A_probe,a_xuvs,a_probes,om_xuvs,om_probes,s_xuvs,s_probes):
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
        
        # Store XUV parameters for ground truth comparison
        self.om0_xuv = om_xuvs[0]

    def define_model(self):

        def rho_peak(x,amp,mu,sigma,a,b,c):
            return amp/(2*sigma) * np.exp(-(x-mu)**2/(2*sigma**2) + 1j*(a*(x-mu)**2 + b*(x-mu) + c))

        def rho_model(e1,e2,amps,mus,sigmas,a_s,bs,cs,Lambdas,gammas):
            retval = 0
            for k in range(len(amps)):
                for l in range(len(amps)):
                    retval += (gammas[k,l] * rho_peak(e1,amps[k],mus[k],sigmas[k],a_s[k],bs[k],cs[k]) 
                               * np.conj(rho_peak(e2,amps[l],mus[l],sigmas[l],a_s[l],bs[l],cs[l])) 
                               * np.exp(-1/2*(Lambdas[k]+Lambdas[l])*(e1-e2)**2))
            trace = sum(
                gammas[k, l].real * amps[k] * amps[l]
                * np.sqrt(2*np.pi / (sigmas[k]**2 + sigmas[l]**2))
                * np.exp(-(mus[k] - mus[l])**2 / (2*(sigmas[k]**2 + sigmas[l]**2)))
                for k in range(len(amps)) for l in range(len(amps))) / 4
            
            return retval/trace
        
        amps = [1.0/np.sqrt(2),1.0]
        mus = [25.0-0.17+self.om_ref*hbar,25.0+self.om_ref*hbar]
        sigmas = [0.1,0.1]
        a_s = [0,4]
        bs = [3,0]
        cs = [0,1]
        Lambdas = [5,5]
        gammas = np.array([[1.0,0.0],
                           [0.0,1.0]])
        
        def rho_f(e1, e2):
            rho = rho_model(e1,e2,amps,mus,sigmas,a_s,bs,cs,Lambdas,gammas)
            return rho 

        self.rho_f = rho_f


    def generate_signal(self,exp_signal=None):
        """Generate synthetic baseline and full signal with optional noise."""

        ### GENERATE SPECTRA

        ir_lo,xuv_lo,ir_hi,xuv_hi = rk.conv_bounds(self.E_lo/hbar,self.E_hi/hbar,self.N_E,self.om_ref)
        self.om_probe = np.linspace(ir_lo,ir_hi,self.N_E)
        self.om_xuv = np.linspace(xuv_lo,xuv_hi,self.N_E)

        ir_lo,xuv_lo,ir_hi,xuv_hi = rk.conv_bounds(self.E_lo/hbar,self.E_hi/hbar,self.N_E_up,self.om_ref)
        self.om_probe_up = np.linspace(ir_lo,ir_hi,self.N_E_up)
        self.om_xuv_up = np.linspace(xuv_lo,xuv_hi,self.N_E_up)

        ref_mask = self.A_ref*((self.om_probe >= self.om_ref - self.s_ref) & (self.om_probe <= self.om_ref + self.s_ref)).astype(float)
        ref_mask_up = self.A_ref*((self.om_probe_up >= self.om_ref - self.s_ref) & (self.om_probe_up <= self.om_ref + self.s_ref)).astype(float)        

        self.sp_probe = rk.sp_tot(self.probes, self.om_probe)
        self.sp_ref = self.sp_probe*ref_mask
        self.sp_xuv = rk.sp_tot(self.xuvs, self.om_xuv)

        self.sp_probe_up = rk.sp_tot(self.probes, self.om_probe_up)
        self.sp_ref_up = self.sp_probe_up*ref_mask_up
        self.sp_xuv_up = rk.sp_tot(self.xuvs, self.om_xuv_up)

        om_probe_reg = rk.regularize_omega(self.om_probe_up)
        om_xuv_reg = rk.regularize_omega(self.om_xuv_up)

        ### PLOT the upsampled spectra with probe/ref on one axis, xuv on another

        plot_spectra(self.om_probe_up,self.om_xuv_up,self.sp_probe_up,self.sp_ref_up,self.sp_xuv_up)
        
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
                caption='Peak sb counts: %i\nN_T: %i\nN_E: %i\nT_res: %.2f fs\nE_res: %.3f eV'%(
                    self.peak_sb_counts,self.N_T,self.N_E,2*self.T_reach/self.N_T,self.E_res),
                saveloc='single_output_temp/pipeline_diag/measured_signal.png',xlabel='Kinetic energy (eV)',ylabel='Time (fs)')
        
        # rk.plot_mat(self.signal_sb, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
        #         # caption='Peak sb counts: %i\nN_T: %i\nN_E: %i\nT_res: %.2f fs\nE_res: %.3f eV'%(
        #         #     self.peak_sb_counts,self.N_T,self.N_E,2*self.T_reach/self.N_T,self.E_res),
        #         saveloc='single_output_temp/pipeline_diag/measured_signal_sb.png',xlabel='Kinetic energy (eV)',ylabel='Time (fs)',title='Sideband signal')
        
    def process_and_detrend(self):
        
        self.signal_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal, use_window=False, zero_pad=self.zero_pad)
        self.signal_sb_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal_sb, use_window=False, zero_pad=self.zero_pad)

        ### PLOT ROIs

        ROI_reach = 2.3
        slice_fracts = (0.5*(1 - ROI_reach/em_hi), 0.5*(1 + ROI_reach/em_hi))
        amplit_tot_FT_ROI1, em_axis_mid, i0, i1 = rk.extract_midslice(self.signal_sb_FT, slice_fracts, hbar*self.OM_T[:,0])
        
        rk.plot_mat(self.signal_FT, extent=[self.E_lo,self.E_hi,em_lo,em_hi],
                saveloc='single_output_temp/pipeline_diag/signal_FT.png', show=False)
        
        rk.plot_mat(np.minimum(amplit_tot_FT_ROI1,4*self.peak_sb_counts), extent=[self.E_lo,self.E_hi,-ROI_reach,ROI_reach],
                saveloc='single_output_temp/pipeline_diag/signal_FT_sb.png',xlabel='Kinetic energy (eV)',ylabel='Indirect energy (eV)', show=False,title='FT of the sideband signal')

        # ### MASK 1-OMEGA REGION
        
        # # Zero-pad the detrended mid-band to the full (ω, E) grid
        # self.sig_sb_FT_zerocomp = np.copy(self.signal_sb_FT)
        
        # # Replace the omega-oscillation feature (positive & negative bands)
        # # with uniform Rician bias from a feature-free donor band.
        # #   mask_lo/mask_hi : indirect-energy band (eV) of the feature (positive side)
        # #   donor_lo        : lower edge (eV) of a feature-free donor band of equal width
        # om_t_eV = hbar * self.OM_T[:, 0]  # indirect energy axis in eV

        # # --- positive band ---
        # mask_i0_pos = np.argmin(np.abs(om_t_eV - self.mask_lo))
        # mask_i1_pos = np.argmin(np.abs(om_t_eV - self.mask_hi))

        # band_width = mask_i1_pos - mask_i0_pos
        # omega_shift = round(1/2*(mask_i0_pos+mask_i1_pos) - self.N_T/2)

        # donor_i0_pos = np.argmin(np.abs(om_t_eV - self.donor_lo))
        # donor_i1_pos = donor_i0_pos + band_width

        # # --- negative band (mirror about 0 eV) ---
        # mask_i0_neg = np.argmin(np.abs(om_t_eV + self.mask_hi))
        # mask_i1_neg = np.argmin(np.abs(om_t_eV + self.mask_lo))

        # donor_i1_neg = np.argmin(np.abs(om_t_eV + self.donor_lo))
        # donor_i0_neg = donor_i1_neg - band_width

        # self.sig_sb_FT_zerocomp[mask_i0_pos:mask_i1_pos, :] = self.signal_sb_FT[donor_i0_pos:donor_i1_pos, :]
        # self.sig_sb_FT_zerocomp[mask_i0_neg:mask_i1_neg, :] = self.signal_sb_FT[donor_i0_neg:donor_i1_neg, :]

        # if (mask_i1_pos + omega_shift) <= self.N_T:
        #     self.sig_sb_FT_zerocomp[mask_i0_pos+omega_shift:mask_i1_pos+omega_shift, :] = self.signal_sb_FT[donor_i0_pos:donor_i1_pos, :]
        #     self.sig_sb_FT_zerocomp[mask_i0_neg-omega_shift:mask_i1_neg-omega_shift, :] = self.signal_sb_FT[donor_i0_neg:donor_i1_neg, :]

        # rk.plot_mat(self.sig_sb_FT_zerocomp, extent=[self.E_lo,self.E_hi,em_lo,em_hi],
        #         saveloc='single_output_temp/pipeline_diag/signal_FT_masked.png', show=False)
        
    def kb_correct(self):
        """Apply Koay-Basser correction for both full and probe signals."""
        # Koay-Basser full signal correction

        tot_rician_full = np.sum(self.signal_sb+self.b, axis=0)

        amp_corr = np.zeros_like(self.signal_sb_FT)
        for j in range(self.N_E):
            col_now = np.abs(self.signal_sb_FT[:,j])
            bias_now = tot_rician_full[j]
            amp_corr[:,j] = rk.koay_basser_correction(col_now, bias_now, lambda_thresh=1.0, only_floor = False)
        phase = np.angle(self.signal_sb_FT)
        
        rk.plot_mat( amp_corr * np.exp(1j*phase) - 1e-5, extent=[self.E_lo,self.E_hi,hbar*self.OM_T[0,0],hbar*self.OM_T[-1,0]],
                saveloc='single_output_temp/pipeline_diag/signal_FT_sb_KBcorr.png', show=False)

        rk.plot_mat(np.abs(self.signal_ft0 -  amp_corr * np.exp(1j*phase) ) / np.max(np.abs(self.signal_ft0)), extent=[self.E_lo,self.E_hi,hbar*self.OM_T[0,0],hbar*self.OM_T[-1,0]],
                saveloc='single_output_temp/pipeline_diag/signal_FT_sb_KBcorr_diff.png', show=False, caption=f'RES = {np.sum(np.abs(self.signal_ft0 - amp_corr * np.exp(1j*phase))) / np.sum(np.abs(self.signal_ft0)):.4f}')

        self.signal_sb_FT = amp_corr * np.exp(1j*phase)


    def probe_sp_correct(self):

        dzeta = 0.001

        probe_modulation = rk.sp_tot(self.probes,self.OM_T)
        where_off = probe_modulation < dzeta
        probe_modulation[where_off] = dzeta

        self.signal_sb_FT_corrected = self.signal_sb_FT / probe_modulation
        self.signal_sb_FT_corrected[where_off] = 0

        self.signal_sb_FT_corrected = median_filter(np.real(self.signal_sb_FT_corrected),size=(3,3)) + 1j*median_filter(np.imag(self.signal_sb_FT_corrected),size=(3,3))

        return

    def resample_analyze(self):
        """Resample and analyze results to produce final density matrices."""
        # Resample and analyze

        rho_rec_unproj, amplit_tot_FT_corrected_small, extent_small, idxs_small, E1, E2 = rk.resample(
            self.signal_sb_FT_corrected, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
        
        rk.plot_mat(rho_rec_unproj, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 saveloc='single_output_temp/rhos/rho_rec_unproj.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='rho_rec_unproj', show=False)
        
        rho_rec = rk.project_to_density_matrix(rho_rec_unproj)

        # Comparison with a ground truth
        ideal_rho = self.rho_f(E1 + self.om_ref*hbar, E2 + self.om_ref*hbar)
        ideal_rho = ideal_rho / np.trace(ideal_rho)
        fid0 = rk.fidelity(ideal_rho, rho_rec)
        print(fid0)

        # FIDELITY NOT APPROACHING EXACTLY 1.0 - HALF-BIN ERRORS FOR IDEAL RHO, PROJECTION WITH LACKING TAILS. OTHERWISE SEEMS PERFECT

        rk.plot_mat(ideal_rho, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 saveloc='single_output_temp/rhos/rho_ideal.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='rho_ideal', show=False, caption='F=1.000')

        rk.plot_mat(rho_rec, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 saveloc='single_output_temp/rhos/rho_rec.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='rho_rec', show=False, caption='F=%.3f'%fid0)

        rk.plot_mat((ideal_rho - rho_rec) / np.max(np.abs(ideal_rho)), extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 saveloc='single_output_temp/rhos/rho_diff.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='rho diff', show=False)
        
    # def WF_reconstruct(self):
    #     """Retrieve probe spectrum using Wirtinger Flow."""

    #     if self.ifWF:
            
    #         sig_probe_reconstructed = self.sig_probe_reconstructed + self.b
            
    #         sp_rec, lasterr = rk.reconstruct_WirtFlow(sig_probe_reconstructed, self.sp_probe, 
    #                                             self.sp_xuv_meas_sig_fit, self.om_probe, self.om_xuv, 
    #                                             self.T, self.b, n_power_iter=50,
    #                                             n_main_iter=3000, ifplot=50, median_regval=4, 
    #                                             eps=self.WF_eps, alph=self.alpha, nt=self.N_T)
            
    #         # Decompose RL reconstruction as a sum of n Gaussians and plot
    #         n_comp = 4
    #         om_grid = self.om_probe
    #         # Use non-negative target for Gaussian fit
            
    #         z_target = np.abs(sp_rec)
            
    #         fit_gauss, fit_params = rk.fit_n_gaussians_1d(
    #             y_vals=om_grid,
    #             z_vals=z_target,
    #             n=n_comp
    #         )

    #         self.probes_reconstructed = rk.nfit_params_to_probes(fit_params, self.T)
            
    #         plt.figure()
    #         # Convert energy (ħ·ω in eV) to wavelength in nm and sort ascending

    #         # E_eV = om_grid * hbar
    #         # lambda_nm = 1239.84197386209 / E_eV
    #         # idx = np.argsort(lambda_nm)[:-30]
            
    #         # plt.plot(lambda_nm[idx], rk.normalize_abs(z_target)[idx], label='WF target')
    #         # plt.plot(lambda_nm[idx], rk.normalize_abs(fit_gauss)[idx], label=f'{n_comp}-Gaussian fit')
    #         # plt.plot(lambda_nm[idx], rk.normalize_abs(self.sp_probe)[idx], label='True spectrum')

    #         plt.plot(om_grid, rk.normalize_abs(z_target), label='WF target')
    #         plt.plot(om_grid, rk.normalize_abs(fit_gauss), label=f'{n_comp}-Gaussian fit')
    #         plt.plot(om_grid, rk.normalize_abs(self.sp_probe), label='True spectrum')

    #         plt.xlabel('lambda [nm] ACTUALLY NO, IT NEEDS FIX')
    #         plt.ylabel('Amplitude (normalized)')
    #         plt.title('n-Gaussian decomposition of RL reconstruction')
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.savefig('single_output_temp/reconstructions/final_sp_probe_rec.png',dpi=300)
    #         plt.close()
        
    #     # === Analytic Gaussian correction (correcting_function_multi) ===
    #     correction = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs, self.om_xuv), 
    #                                          rk.normalize_params(self.probes, self.om_probe), 
    #                                          dzeta=self.dzeta_val, theta=self.theta_val)
    #     self.amplit_tot_FT_corrected = correction*self.signal_sb_FT
    #     self.amplit_tot_FT_corrected = median_filter(np.abs(self.amplit_tot_FT_corrected), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected))
        
    #     correction_x = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs_rec, self.om_xuv), 
    #                                          rk.normalize_params(self.probes, self.om_probe), 
    #                                          dzeta=self.dzeta_val, theta=self.theta_val)
    #     self.amplit_tot_FT_corrected_x = correction_x*self.signal_sb_FT
    #     self.amplit_tot_FT_corrected_x = median_filter(np.abs(self.amplit_tot_FT_corrected_x), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected_x))
        
    #     if self.ifWF:
    #         if self.ifexp:
    #             correction_rec_x = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs, self.om_xuv), 
    #                                                 rk.normalize_params(self.probes_reconstructed, self.om_probe), 
    #                                                 dzeta=self.dzeta_val, theta=self.theta_val)
    #             self.amplit_tot_FT_corrected_rec_x_nomedian = correction_rec_x*self.signal_sb_FT
    #             self.amplit_tot_FT_corrected_rec_x = median_filter(np.abs(self.amplit_tot_FT_corrected_rec_x_nomedian), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected_rec_x_nomedian))
    #         else:
    #             correction_rec_x = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs_rec, self.om_xuv), 
    #                                                 rk.normalize_params(self.probes_reconstructed, self.om_probe), 
    #                                                 dzeta=self.dzeta_val, theta=self.theta_val)
    #             self.amplit_tot_FT_corrected_rec_x_nomedian = correction_rec_x*self.signal_sb_FT
    #             self.amplit_tot_FT_corrected_rec_x = median_filter(np.abs(self.amplit_tot_FT_corrected_rec_x_nomedian), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected_rec_x_nomedian))


    # def xuv_peak(self):
    #     """Reconstruct XUV peak spectrum."""

    #     xuv_sigq_avg = np.sum(self.signal_harmq,axis=0)
    #     xuv_sigq_avg = rk.normalize_abs(np.abs(xuv_sigq_avg).astype(float))

    #     if self.if_coherent:
    #         xuv_sigq_avg_fitted, fit_params = rk.fit_n_gaussians_1d(self.E[0,:]/hbar,np.sqrt(xuv_sigq_avg),2,saveloc='single_output_temp/reconstructions/harmq_fit.png',plotlim=[self.harmq_lo/hbar,self.harmq_hi/hbar])
    #         self.xuvs_rec = rk.nfit_params_to_probes(fit_params, 0*self.T)
    #     else:
    #         xuv_sigq_avg_fitted, fit_params = rk.fit_n_gaussians_1d(self.E[0,:]/hbar,xuv_sigq_avg,2,saveloc='single_output_temp/reconstructions/harmq_fit.png',plotlim=[self.harmq_lo/hbar,self.harmq_hi/hbar])
    #         self.xuvs_rec = rk.sqrt_probes(rk.nfit_params_to_probes(fit_params, 0*self.T))
    #     sp_xuv_rec = rk.sp_tot(self.xuvs_rec,self.om_xuv)

    #     plt.plot(self.om_xuv*hbar,rk.normalize_abs(self.sp_xuv),label='measured xuv band')
    #     plt.plot(self.om_xuv*hbar,rk.normalize_abs(sp_xuv_rec),label='fitted xuv band')
    #     plt.xlim([self.harmq_lo-0.2,self.harmq_hi+0.2])
    #     plt.xlabel('Energy [eV]')
    #     plt.ylabel('Amplitude (normalized)')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig('single_output_temp/reconstructions/sp_xuv_rec.png',dpi=300)
    #     plt.close()

    #     if self.ifexp:

    #         p1 = rk.normalize_abs(xuv_sigq_avg)
    #         p2 = rk.normalize_abs(rk.sp_tot((self.xuvs[0],),self.E[0,:]/hbar)**2 + rk.sp_tot((self.xuvs[1],),self.E[0,:]/hbar)**2)

    #         plt.plot(self.E[0,:],p1,label='average mainband')
    #         plt.plot(self.E[0,:],p2*np.max(p1)/np.max(p2),label='mainband from hand-tuned \nxuv spectrum')
    #         plt.xlim([self.harmq_lo-0.2,self.harmq_hi+0.2])
    #         plt.xlabel('Energy [eV]')
    #         plt.ylabel('Amplitude (normalized)')
    #         plt.title('XUV spectrum tuning')
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.savefig('single_output_temp/reconstructions/harmq_compare.png',dpi=300)
    #         plt.close()
        
    #     self.sp_xuv_meas_sig_fit = rk.sp_tot(self.xuvs_rec,self.om_xuv)
        
    # def resample_analyze(self):
    #     """Resample and analyze results to produce final density matrices."""
    #     # Resample and analyze

    #     rho_uncorrected, amplit_tot_FT_corrected_small, extent_small, idxs_small, _, _ = rk.resample(
    #         median_filter(np.abs(self.signal_sb_FT),size=(3,3))*np.exp(1j*np.angle(self.signal_sb_FT)), self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
        
    #     rk.plot_mat(rho_uncorrected, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
    #              mode='abs', saveloc='single_output_temp/rhos/synth_uncorr_unproj.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
    #              title='Rho uncorrected for the probe spectrum', show=False)
        
    #     rho_uncorrected = rk.project_to_density_matrix(1/2*(rho_uncorrected+rho_uncorrected.T))

    #     rho_reconstructed, amplit_tot_FT_corrected_small, extent_small, idxs_small, E1, E2 = rk.resample(
    #         self.amplit_tot_FT_corrected, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
    #     rho_reconstructed = rk.project_to_density_matrix(rho_reconstructed)
        
    #     rho_reconstructed_x, amplit_tot_FT_corrected_small, extent_small, idxs_small, _, _ = rk.resample(
    #         self.amplit_tot_FT_corrected_x, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
    #     rho_reconstructed_x = rk.project_to_density_matrix(rho_reconstructed_x)
        
    #     if self.ifWF:
    #         rho_reconstructed_rec_x, amplit_tot_FT_corrected_small_rec, extent_small, idxs_small, _, _ = rk.resample(
    #             self.amplit_tot_FT_corrected_rec_x, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
    #         rho_reconstructed_rec_x = rk.project_to_density_matrix(rho_reconstructed_rec_x)
            
    #         rho_reconstructed_rec_x_nomedian, amplit_tot_FT_corrected_small_rec, extent_small, idxs_small, _, _ = rk.resample(
    #             self.amplit_tot_FT_corrected_rec_x_nomedian, self.rho_hi, self.rho_lo, self.om_ref, self.E, self.OM_T, self.N_T)
    #         rho_reconstructed_rec_x_nomedian = rk.project_to_density_matrix(rho_reconstructed_rec_x_nomedian)
        
    #     # Comparison with a ground truth

    #     plt.plot(E1[:,0],rk.sp_tot(self.xuvs, E1[:,0]/hbar))
    #     plt.savefig('fig1.png')
    #     plt.close()

    #     if self.if_coherent:
    #         ### Ground truth - coherent sum
    #         xuv_gt = rk.sp_tot(self.xuvs, E1[:,0]/hbar)
    #         ideal_rho = xuv_gt[:,np.newaxis]@np.conjugate(xuv_gt[np.newaxis,:])
    #         ideal_rho = rk.project_to_density_matrix(ideal_rho, smooth_sigma=0.1)
    #     else:
    #         ### Ground truth - incoherent sum of individual XUV component density matrices
    #         ideal_rho = np.zeros((len(E1[:,0]), len(E1[:,0])), dtype=complex)
    #         for xuv_i in self.xuvs:
    #             if self.ifexp:
    #                 xuv_gt_i = rk.sp_tot((xuv_i,), (E1[:,0]-0.06)/hbar)
    #             else:
    #                 xuv_gt_i = rk.sp_tot((xuv_i,), E1[:,0]/hbar)
    #             ideal_rho += xuv_gt_i[:,np.newaxis]@np.conjugate(xuv_gt_i[np.newaxis,:])
    #         ideal_rho = rk.project_to_density_matrix(ideal_rho, smooth_sigma=0.1)

    #     fid0 = rk.fidelity(ideal_rho, rho_uncorrected)
    #     fid1 = rk.fidelity(ideal_rho, rho_reconstructed)
    #     fid2 = rk.fidelity(ideal_rho, rho_reconstructed_x)
    #     if self.ifWF:
    #         fid3 = rk.fidelity(ideal_rho, rho_reconstructed_rec_x)
    #         fid4 = rk.fidelity(ideal_rho, rho_reconstructed_rec_x_nomedian)
    #         print(fid0,fid1,fid2,fid3,fid4)
    #     else:
    #         print(fid0,fid1,fid2)
        
    #     rk.plot_mat(ideal_rho, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
    #              mode='abs', saveloc='single_output_temp/rhos/ideal_rho.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
    #              title='True quantum state', show=False, caption='F=1.00')
        
    #     rk.plot_mat(rho_uncorrected, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
    #              mode='abs', saveloc='single_output_temp/rhos/synth_uncorr.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
    #              title='Rho uncorrected for the probe spectrum', show=False, caption='F=%.3f'%fid0)
        
    #     rk.plot_mat(rho_reconstructed, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
    #              mode='abs', saveloc='single_output_temp/rhos/synth_corr.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
    #              title='Rho ideally corrected for the probe spectrum', show=False, caption='F=%.3f'%fid1)
        
    #     rk.plot_mat(rho_reconstructed_x, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
    #              mode='abs', saveloc='single_output_temp/rhos/synth_corr_x.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
    #              title='Rho rec xuv and ideally corrected for the probe spectrum', show=False, caption='F=%.3f'%fid2)

    #     if self.ifWF:
    #         rk.plot_mat(rho_reconstructed_rec_x, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
    #                 mode='phase', saveloc='single_output_temp/rhos/rec_corr_x_phase.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
    #                 title='Rho rec xuv and WF rec probe', show=False, caption='F=%.3f'%fid3)
    #         rk.plot_mat(rho_reconstructed_rec_x, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
    #                 mode='abs', saveloc='single_output_temp/rhos/rec_corr_x.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
    #                 title='Rho rec xuv and WF rec probe', show=False, caption='F=%.3f'%fid3)
            
    #         rk.plot_mat(rho_reconstructed_rec_x_nomedian, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
    #                 mode='abs', saveloc='single_output_temp/rhos/rec_corr_x_nomedian.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
    #                 title='Rho rec xuv and WF rec probe', show=False, caption='F=%.3f'%fid4)

    #     # Store results as instance attributes
    #     self.ideal_rho = ideal_rho
    #     self.rho_uncorrected = rho_uncorrected
    #     self.rho_reconstructed = rho_reconstructed
    #     self.rho_reconstructed_x = rho_reconstructed_x
    #     if self.ifWF:
    #         self.rho_reconstructed_rec_x = rho_reconstructed_rec_x
    #         self.rho_reconstructed_rec_x_nomedian = rho_reconstructed_rec_x_nomedian
    #         self.fidelities = [fid0, fid1, fid2, fid3, fid4]
    #     else:
    #         self.fidelities = [fid0, fid1, fid2]


if __name__ == "__main__":

    E_lo = 24.5
    # E_hi = 29
    E_hi = 28.0
    T_reach = 100
    # T_reach = 250
    # E_res = 0.01    
    E_res = 0.01    
    N_T = 501
    # N_T = 700
    p_E = 4  # N_E upsampling integer
    alpha = 10000
    b = 1

    sideband_lo = 25.5
    sideband_hi = 28.0
    harmq_lo = 24.5
    harmq_hi = 25.5

    if_coherent = True

    # Define pulses

    scale = 0.5

    A_xuv = 0.1*scale
    a_xuvs = [0.8]
    om_xuvs = [25.00/hbar]
    s_xuvs = [0.15/hbar]

    # n_max = 8
    # A_xuv = 0.1*scale
    # a_xuvs = [0.8 for n in range(2*n_max)]
    # om_xuvs = [(n+1) * 1.50/hbar if n%2==0 else n * 1.50/hbar + 0.3/hbar for n in range(2*n_max)]
    # s_xuvs = [0.10/hbar for n in range(2*n_max)]

    A_probe = 1.2*scale
    a_probes = [1.0,0.05,0.05]
    om_probes = [1.55/hbar,1.51/hbar,1.58/hbar]
    s_probes = [0.05/hbar,0.02/hbar,0.02/hbar]

    # # Flat-top probe (1.45–1.60 eV, 31 Gaussians, box approx)
    # A_probe = 0.1
    # a_probes = [0.35]*30
    # om_probes = [1.40/hbar + i*0.005/hbar for i in range(30)]
    # s_probes = [0.005/hbar]*30
    # Flat-top probe (1.45–1.60 eV, 31 Gaussians, box approx)

    # A_probe = 0.5
    # a_probes = [1.0]
    # om_probes = [1.50/hbar]
    # s_probes = [0.10/hbar]

    A_ref = 1.0
    om_ref = 1.48/hbar
    s_ref = 0.015/hbar

    # Create experiment instance
    experiment = RK_experiment(E_lo=E_lo,E_hi=E_hi,T_reach=T_reach,E_res=E_res,N_T=N_T,p_E=p_E,alpha=alpha,b=b,
                            sb_lo=sideband_lo,sb_hi=sideband_hi,harmq_lo=harmq_lo,harmq_hi=harmq_hi,if_coherent=if_coherent,
                            A_ref=A_ref,om_ref=om_ref,s_ref=s_ref)

    experiment.ifWF = False        

    experiment.define_pulses(A_xuv,A_probe,a_xuvs,a_probes,om_xuvs,om_probes,s_xuvs,s_probes)
    experiment.define_model()
    experiment.generate_signal()
    experiment.process_and_detrend()
    experiment.kb_correct()
    experiment.probe_sp_correct()
    experiment.resample_analyze()