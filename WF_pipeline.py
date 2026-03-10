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
    
    def __init__(self,E_lo=60.0,E_hi=63.5,T_reach=50,E_res=0.025,N_T=240,p_E=4,alpha=0.05,b=1,sb_lo=24.7,sb_hi=28,harmq_lo=22,harmq_hi=24.7,
                 if_coherent=True,mask_lo=0.8,mask_hi=2.2,donor_lo=2.5,E_range=None):
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
        self.ifnoise = False
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
        self.dzeta_val = 1e-3
        self.theta_val = 0.01
        
        # Band parameters
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
        self.om0_ref = om_refs[0]  # Use first reference frequency
        
        # Store XUV parameters for ground truth comparison
        self.om0_xuv = om_xuvs[0]
            
    def generate_signal(self,exp_signal=None):
        """Generate synthetic baseline and full signal with optional noise."""

        # om_xuv = np.linspace(self.om0_xuv-self.E_span/hbar/2,self.om0_xuv+self.E_span/hbar/2,self.N_E_up)
        # om_ir = np.linspace(-self.E_span/hbar/2,+self.E_span/hbar/2,self.N_E_up)
        # om_ir_reg = rk.regularize_omega(om_ir)

        # sp_xuv = rk.sp_tot(self.xuvs,om_xuv) * np.exp(1j*0*self.T_up)
        # # sp_ref = rk.sp_tot(self.refs,om_ir) * np.exp(1j*0*self.T_up) + rk.sp_tot(rk.pulse_emit(self.refs),om_ir) * np.exp(1j*0*self.T_up)
        # sp_ref = rk.sp_tot(self.refs,om_ir) * np.exp(1j*0*om_ir*self.T_up/2) + rk.sp_tot(rk.pulse_emit(self.refs),om_ir) * np.exp(-1j*0*om_ir*self.T_up/2)
        # sp_probe = rk.sp_tot(self.probes,om_ir) * np.exp(1j*om_ir*self.T_up) + rk.sp_tot(rk.pulse_emit(self.probes),om_ir) * np.exp(-1j*om_ir*self.T_up)

        # plt.plot(np.abs(sp_probe[0,:])**2)
        # plt.savefig('fig0.png')
        # plt.close()

        # def cut_axes(conv,om1,om2,lo):
        #     conv_ax = np.linspace(om1[0]+om2[0],om1[-1]+om2[-1],np.size(om1)+np.size(om2)-1)
        #     i_lo = np.argmin(np.abs(conv_ax - lo))
        #     i_hi = i_lo + np.size(om1)
        #     return conv[:,i_lo:i_hi]
        
        # d_om = om_ir[1] - om_ir[0]

        # A1 = rk.sp_tot(self.xuvs,self.E_up[0,:]/hbar) * np.exp(1j*0*self.T_up)

        # A2_xi = d_om*(fftconvolve(sp_ref,-sp_xuv/om_xuv,axes=1) + fftconvolve(sp_xuv,-sp_ref/om_ir_reg,axes=1) +
        #              fftconvolve(sp_probe,-sp_xuv/om_xuv,axes=1) + fftconvolve(sp_xuv,-sp_probe/om_ir_reg,axes=1))
        # A2_xi_ax = rk.regularize_omega(np.linspace(om_xuv[0]+om_ir[0],om_xuv[-1]+om_ir[-1],2*self.N_E_up-1))

        # A2_ii = d_om*(fftconvolve(sp_ref,-sp_probe/om_ir_reg,axes=1) + fftconvolve(sp_probe,-sp_ref/om_ir_reg,axes=1))
        # A2_ii_ax = rk.regularize_omega(np.linspace(2*om_ir[0],2*om_ir[-1],2*self.N_E_up-1))

        # A3_i_xi = d_om*(fftconvolve(sp_ref,-A2_xi/A2_xi_ax,axes=1) + fftconvolve(A2_xi,-sp_ref/om_ir_reg,axes=1) + 
        #                 fftconvolve(sp_probe,-A2_xi/A2_xi_ax,axes=1) + fftconvolve(A2_xi,-sp_probe/om_ir_reg,axes=1))

        # A3_x_ii = d_om*(fftconvolve(sp_xuv,-A2_ii/A2_ii_ax,axes=1) + fftconvolve(A2_ii,-sp_xuv/om_xuv,axes=1))

        # A_tot = A1 + cut_axes(A2_xi,om_ir,om_xuv,self.E_lo/hbar) + cut_axes(A3_i_xi,om_ir,A2_xi_ax,self.E_lo/hbar) + cut_axes(A3_x_ii,om_xuv,A2_ii_ax,self.E_lo/hbar)

        # S = np.abs(A_tot)**2

        # rk.plot_mat(A2_ii, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
        #         saveloc='figg.png',xlabel='Kinetic energy (eV)',ylabel='Time (fs)')

        # rk.plot_mat(np.minimum(S,0.0006), extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
        #         saveloc='single_output_temp/pipeline_diag/measured_signal.png',xlabel='Kinetic energy (eV)',ylabel='Time (fs)')

        # self.signal_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, S, use_window=False, zero_pad=self.zero_pad)
        
        # rk.plot_mat(np.minimum(np.abs(self.signal_FT),0.0010)*np.exp(1j*np.angle(self.signal_FT)), extent=[self.E_lo,self.E_hi,em_lo,em_hi],
        #         saveloc='single_output_temp/pipeline_diag/raw_signal_FT.png', show=False)


        # # plt.plot(np.sum(S[:,320:480],axis=1))
        # plt.plot(2*np.sum(S[:,:320],axis=1))
        # plt.plot(np.sum(S,axis=1))
        # # plt.ylim([0,0.12])
        # plt.savefig('fig1.png')
        # exit()

        #########

        ir_lo,xuv_lo,ir_hi,xuv_hi = rk.conv_bounds(self.E_lo/hbar,self.E_hi/hbar,self.N_E,self.om0_ref)
        self.om_probe = np.linspace(ir_lo,ir_hi,self.N_E)
        self.om_xuv = np.linspace(xuv_lo,xuv_hi,self.N_E)
        
        ir_lo,xuv_lo,ir_hi,xuv_hi = rk.conv_bounds(self.E_lo/hbar,self.E_hi/hbar,self.N_E_up,self.om0_ref)
        self.om_probe_up = np.linspace(ir_lo,ir_hi,self.N_E_up)
        self.om_xuv_up = np.linspace(xuv_lo,xuv_hi,self.N_E_up)

        self.sp_probe = rk.sp_tot(self.probes, self.om_probe)
        self.sp_ref = rk.sp_tot(self.refs, self.om_probe)
        self.sp_xuv = rk.sp_tot(self.xuvs, self.om_xuv)

        self.sp_probe_up = rk.sp_tot(self.probes, self.om_probe_up)
        self.sp_ref_up = rk.sp_tot(self.refs, self.om_probe_up)
        self.sp_xuv_up = rk.sp_tot(self.xuvs, self.om_xuv_up)

        ## Emit

        ir_lo,xuv_lo,ir_hi,xuv_hi = rk.conv_bounds(self.E_lo/hbar,self.E_hi/hbar,self.N_E_up,-self.om0_ref)

        self.om_probe_up_emit = np.linspace(ir_lo,ir_hi,self.N_E_up)
        self.om_xuv_up_emit = np.linspace(xuv_lo,xuv_hi,self.N_E_up)

        self.sp_xuv_up_emit = rk.sp_tot(self.xuvs, self.om_xuv_up_emit)
        self.sp_probe_up_emit = rk.sp_tot(rk.pulse_emit(self.probes), self.om_probe_up_emit)
        self.sp_ref_up_emit = rk.sp_tot(rk.pulse_emit(self.refs), self.om_probe_up_emit)

        om_probe_reg = rk.regularize_omega(self.om_probe_up)
        om_xuv_reg = rk.regularize_omega(self.om_xuv_up)
        
        om_probe_emit_reg = rk.regularize_omega(self.om_probe_up_emit)
        om_xuv_emit_reg = rk.regularize_omega(self.om_xuv_up_emit)

        ## Plot the upsampled spectra with probe/ref on one axis, xuv on another

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
                # Second-order amplitudes
                A2_probe = rk.synth_baseline_n(self.sp_probe_up, self.sp_xuv_up, 
                                             om_probe_reg, om_xuv_reg, self.T_up)
                A2_ref = rk.synth_baseline_n(self.sp_ref_up, self.sp_xuv_up, 
                                           om_probe_reg, om_xuv_reg, self.T_up*0)
                A2_probe_emit = rk.synth_baseline_n(self.sp_probe_up_emit, self.sp_xuv_up_emit, 
                                             om_probe_emit_reg, om_xuv_emit_reg, self.T_up)
                A2_ref_emit = rk.synth_baseline_n(self.sp_ref_up_emit, self.sp_xuv_up_emit, 
                                           om_probe_emit_reg, om_xuv_emit_reg, self.T_up*0)
                A2_total = A2_probe + A2_ref + A2_probe_emit + A2_ref_emit

                # Third-order: convolve total second-order with each IR field

                phase0 = np.exp(1j*0*self.T_up)
                
                phase1 = np.exp(1j*self.om_probe_up*self.T_up)

                sp_tau_ref = self.sp_ref_up * phase0
                sp_tau_probe = self.sp_probe_up * phase1

                phase1 = np.exp(1j*self.om_probe_up_emit*self.T_up)
                sp_tau_ref_emit = self.sp_ref_up_emit * phase0
                sp_tau_probe_emit = self.sp_probe_up_emit * phase1

                prop = -1/(om_probe_reg*phase0)
                prop_emit = -1/(om_probe_emit_reg*phase0)

                # Slice the full convolution so A3 lives on the same
                # frequency grid as A2_total (analogous to mode='same' in
                # synth_baseline, but with the correct offset for the
                # probe frequency axis).
                d_om = self.om_probe_up[1] - self.om_probe_up[0]
                n0 = round(-self.om_probe_up[0] / d_om)
                N_up = self.N_E_up

                A3_probe = rk.synth_baseline_n(self.sp_probe_up,A2_total,om_probe_reg,self.E_up[0,:]/hbar,self.T_up,mode='full')[:, n0:n0+N_up] #fftconvolve(sp_tau_probe, A2_total*prop, axes=1)[:, n0:n0+N_up]*d_om
                A3_ref = rk.synth_baseline_n(self.sp_ref_up,A2_total,om_probe_reg,self.E_up[0,:]/hbar,self.T_up*0,mode='full')[:, n0:n0+N_up] #fftconvolve(sp_tau_ref, A2_total*prop, axes=1)[:, n0:n0+N_up]*d_om

                n0_emit = round(-self.om_probe_up_emit[0] / d_om)
                A3_probe_emit = rk.synth_baseline_n(self.sp_probe_up_emit,A2_total,om_probe_reg,self.E_up[0,:]/hbar,self.T_up,mode='full')[:, n0_emit:n0_emit+N_up] #fftconvolve(sp_tau_probe_emit, A2_total*prop_emit, axes=1)[:, n0_emit:n0_emit+N_up]*d_om
                A3_ref_emit = rk.synth_baseline_n(self.sp_ref_up_emit,A2_total,om_probe_reg,self.E_up[0,:]/hbar,self.T_up*0,mode='full')[:, n0_emit:n0_emit+N_up] #fftconvolve(sp_tau_ref_emit, A2_total*prop_emit, axes=1)[:, n0_emit:n0_emit+N_up]*d_om

                A3_total = A3_probe + A3_ref + A3_probe_emit + A3_ref_emit

                rk.plot_mat(A3_probe,saveloc='figg.png')
                rk.plot_mat(A3_probe_emit,saveloc='figg1.png')

                amplit_tot_0 = rk.downsample(
                                        A2_total
                                        + sp_xuv_E
                                        + A3_total
                                        , self.p_E)
                signal_clean = np.abs(amplit_tot_0)**2
            else:
                # Incoherent: sum |amplitude|^2 from each XUV component
                signal_clean = np.zeros((self.N_T, self.N_E))
                for xuv_i in self.xuvs:
                    sp_xuv_i = rk.sp_tot((xuv_i,), self.om_xuv_up)
                    sp_xuv_i_E = rk.sp_tot((xuv_i,), self.E_up/hbar)
                    # Second-order amplitudes
                    A2_probe_i = rk.synth_baseline(self.sp_probe_up, sp_xuv_i, 
                                                   om_probe_reg, om_xuv_reg, self.T_up)
                    A2_ref_i = rk.synth_baseline(self.sp_ref_up, sp_xuv_i, 
                                                 om_probe_reg, om_xuv_reg, self.T_up*0)
                    A2_total_i = A2_probe_i + A2_ref_i
                    # Third-order: convolve total second-order with each IR field
                    A3_probe_i = rk.synth_baseline(self.sp_probe_up, A2_total_i, 
                                                   om_probe_reg, om_xuv_reg, self.T_up)
                    A3_ref_i = rk.synth_baseline(self.sp_ref_up, A2_total_i, 
                                                 om_probe_reg, om_xuv_reg, self.T_up*0)
                    amplit_i = rk.downsample(
                                        A2_total_i
                                        + A3_probe_i + A3_ref_i
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
                    
            self.signal -= self.b # NOISE FLOOR SHOULD BE QUITE PRECISELY MEASURED BY CAPTURING WITH LASER OFF

            check = np.sum(self.signal,axis=1)
            plt.plot(check)
            plt.savefig('fig1.png')

            # Zero out signal outside the sideband range [sb_lo, sb_hi]
            sb_lo_idx = np.argmin(np.abs(self.E_range - self.sb_lo))
            sb_hi_idx = np.argmin(np.abs(self.E_range - self.sb_hi))
            self.signal_sb = np.zeros_like(self.signal)
            self.signal_sb[:, sb_lo_idx:sb_hi_idx+1] = self.signal[:, sb_lo_idx:sb_hi_idx+1]

            harmq_lo_idx = np.argmin(np.abs(self.E_range - self.harmq_lo))
            harmq_hi_idx = np.argmin(np.abs(self.E_range - self.harmq_hi))
            self.signal_harmq = np.zeros_like(self.signal)
            self.signal_harmq[:, harmq_lo_idx:harmq_hi_idx+1] = self.signal[:, harmq_lo_idx:harmq_hi_idx+1]

            peak_sb_counts = int(np.max(self.signal_sb))
            self.peak_sb_counts = peak_sb_counts

        else:
            self.signal = exp_signal

            # # Deconvolve each row of the signal with the previously saved broadening kernel
            # kernel_path = 'single_output_temp/reconstructions/h_kernel.npy'
            # try:
            #     h_kernel = np.load(kernel_path)
            #     # Richardson-Lucy deconvolution per row (axis=1, i.e. along E)
            #     n_rl_iter = 50
            #     eps_rl = 1e-12
            #     h_flip = h_kernel[::-1]
            #     deconv_signal = np.maximum(self.signal, 0.0).copy()
            #     for _rl_i in range(n_rl_iter):
            #         conv_fwd = np.apply_along_axis(
            #             lambda row: fftconvolve(row, h_kernel, mode='same'), 1, deconv_signal)
            #         ratio = np.maximum(self.signal, 0.0) / (conv_fwd + eps_rl)
            #         correction = np.apply_along_axis(
            #             lambda row: fftconvolve(row, h_flip, mode='same'), 1, ratio)
            #         deconv_signal *= correction
            #         deconv_signal = np.maximum(deconv_signal, 0.0)
            #         print(_rl_i)
            #     # Rescale each row so total counts are preserved
            #     row_sums_orig = np.maximum(self.signal, 0.0).sum(axis=1, keepdims=True)
            #     row_sums_deconv = deconv_signal.sum(axis=1, keepdims=True)
            #     scale = np.where(row_sums_deconv > 0, row_sums_orig / row_sums_deconv, 1.0)
            #     deconv_signal *= scale
            #     self.signal = deconv_signal
            #     print(f'Deconvolved exp_signal with kernel from {kernel_path}')
            # except FileNotFoundError:
            #     print(f'WARNING: kernel file {kernel_path} not found — skipping deconvolution')

            # Zero out signal outside the sideband range [sb_lo, sb_hi]
            sb_lo_idx = np.argmin(np.abs(self.E_range - self.sb_lo))
            sb_hi_idx = np.argmin(np.abs(self.E_range - self.sb_hi))
            self.signal_sb = np.zeros_like(self.signal)
            self.signal_sb[:, sb_lo_idx:sb_hi_idx+1] = self.signal[:, sb_lo_idx:sb_hi_idx+1]

            harmq_lo_idx = np.argmin(np.abs(self.E_range - self.harmq_lo))
            harmq_hi_idx = np.argmin(np.abs(self.E_range - self.harmq_hi))
            self.signal_harmq = np.zeros_like(self.signal)
            self.signal_harmq[:, harmq_lo_idx:harmq_hi_idx+1] = self.signal[:, harmq_lo_idx:harmq_hi_idx+1]

            peak_sb_counts = int(np.max(self.signal_sb))
            self.peak_sb_counts = peak_sb_counts

            # self.xuv_peak()

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
    
    def process_and_detrend(self):
        
        self.signal_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal, use_window=False, zero_pad=self.zero_pad)
        self.signal_sb_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal_sb, use_window=True, zero_pad=self.zero_pad)
        self.signal_detrend_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal_sb, use_window=self.window_zerocomp, zero_pad=self.zero_pad)
        
        rk.plot_mat(np.minimum(np.abs(self.signal_FT),100)*np.exp(1j*np.angle(self.signal_FT)), extent=[self.E_lo,self.E_hi,em_lo,em_hi],
                saveloc='single_output_temp/pipeline_diag/signal_FT.png', show=False)
        
        ###
        ### PLOT ROIs
        ###

        ROI_reach = 2.3
        slice_fracts = (0.5*(1 - ROI_reach/em_hi), 0.5*(1 + ROI_reach/em_hi))
        
        amplit_tot_FT_ROI, em_axis_mid, i0, i1 = rk.extract_midslice(self.signal_detrend_FT, slice_fracts, hbar*self.OM_T[:,0])
        
        rk.plot_mat(np.minimum(amplit_tot_FT_ROI,4*self.peak_sb_counts), extent=[self.E_lo,self.E_hi,-ROI_reach,ROI_reach],
                saveloc='single_output_temp/pipeline_diag/signal_sb_FT.png',xlabel='Kinetic energy (eV)',ylabel='Indirect energy (eV)', show=False,title='FT of the sideband signal')
        
        ROI_reach_lo = 1.0
        ROI_reach_hi = 2.2
        slice_fracts = (0.5*(1 + ROI_reach_lo/em_hi), 0.5*(1 + ROI_reach_hi/em_hi))
        ROI_e_reach_lo = self.sb_lo
        ROI_e_reach_hi = self.sb_hi
        slice_e_fracts = ((ROI_e_reach_lo - self.E_lo) / (self.E_hi - self.E_lo), 
                          (ROI_e_reach_hi - self.E_lo) / (self.E_hi - self.E_lo))
        
        amplit_tot_FT_ROI, em_axis_mid, i0, i1, e_axis_mid, e_i0, e_i1 = rk.extract_midslice(self.signal_detrend_FT, slice_fracts, hbar*self.OM_T[:,0],
                                                                                             e_slice_fracts=slice_e_fracts, e_sliced_range=self.E[0,:])
        
        rk.plot_mat(np.minimum(amplit_tot_FT_ROI,4*self.peak_sb_counts), extent=[ROI_e_reach_lo,ROI_e_reach_hi,ROI_reach_lo,ROI_reach_hi],
                saveloc='single_output_temp/pipeline_diag/signal_FT_ROI.png', show=False)
        
        ###
        ###
        ###
        
        # Detrending - separating probe and ref zero freq component
        slice_fracts = (0.5*(1 - self.em_axis_mid_reach/em_hi), 0.5*(1 + self.em_axis_mid_reach/em_hi))
        
        sig_sb_FT_mid, em_axis_mid, i0, i1 = rk.extract_midslice(self.signal_detrend_FT, slice_fracts, hbar*self.OM_T[:,0])
        
        rk.plot_mat(sig_sb_FT_mid, extent=[self.E_lo,self.E_hi,-self.em_axis_mid_reach,self.em_axis_mid_reach],
                saveloc='single_output_temp/pipeline_diag/signal_sb_FT_mid.png', show=False)
        
        sig_sb_FT_mid_detrended, spike_only, spike_row_mid = rk.detrend_spike(sig_sb_FT_mid, em_axis_mid, 0, 2, N_E=self.N_E, plot=False)
        
        rk.plot_mat(sig_sb_FT_mid_detrended, extent=[self.E_lo,self.E_hi,-self.em_axis_mid_reach,self.em_axis_mid_reach],
                saveloc='single_output_temp/pipeline_diag/signal_sb_FT_mid_detrended.png', show=False)
        
        # Zero-pad the detrended mid-band to the full (ω, E) grid
        self.sig_sb_FT_full_detrended = np.copy(self.signal_detrend_FT)
        self.sig_sb_FT_full_detrended[i0:i1, :] = sig_sb_FT_mid_detrended
        
        # Replace the omega-oscillation feature (positive & negative bands)
        # with uniform Rician bias from a feature-free donor band.
        #   mask_lo/mask_hi : indirect-energy band (eV) of the feature (positive side)
        #   donor_lo        : lower edge (eV) of a feature-free donor band of equal width
        om_t_eV = hbar * self.OM_T[:, 0]  # indirect energy axis in eV

        # --- positive band ---
        mask_i0_pos = np.argmin(np.abs(om_t_eV - self.mask_lo))
        mask_i1_pos = np.argmin(np.abs(om_t_eV - self.mask_hi))

        band_width = mask_i1_pos - mask_i0_pos
        omega_shift = round(1/2*(mask_i0_pos+mask_i1_pos) - self.N_T/2)

        donor_i0_pos = np.argmin(np.abs(om_t_eV - self.donor_lo))
        donor_i1_pos = donor_i0_pos + band_width

        # --- negative band (mirror about 0 eV) ---
        mask_i0_neg = np.argmin(np.abs(om_t_eV + self.mask_hi))
        mask_i1_neg = np.argmin(np.abs(om_t_eV + self.mask_lo))

        donor_i1_neg = np.argmin(np.abs(om_t_eV + self.donor_lo))
        donor_i0_neg = donor_i1_neg - band_width

        self.sig_sb_FT_full_detrended[mask_i0_pos:mask_i1_pos, :] = self.signal_detrend_FT[donor_i0_pos:donor_i1_pos, :]
        self.sig_sb_FT_full_detrended[mask_i0_neg:mask_i1_neg, :] = self.signal_detrend_FT[donor_i0_neg:donor_i1_neg, :]

        if (mask_i1_pos + omega_shift) <= self.N_T:
            self.sig_sb_FT_full_detrended[mask_i0_pos+omega_shift:mask_i1_pos+omega_shift, :] = self.signal_detrend_FT[donor_i0_pos:donor_i1_pos, :]
            self.sig_sb_FT_full_detrended[mask_i0_neg-omega_shift:mask_i1_neg-omega_shift, :] = self.signal_detrend_FT[donor_i0_neg:donor_i1_neg, :]

        rk.plot_mat(self.sig_sb_FT_full_detrended, extent=[self.E_lo,self.E_hi,em_lo,em_hi],
                saveloc='single_output_temp/pipeline_diag/signal_sb_FT_full_detrended.png', show=False)
        
        self.xuv_peak()

    
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
        
        amp_corr = np.zeros_like(self.signal_sb_FT)
        
        for j in range(self.N_E):
            col_now = np.abs(self.signal_sb_FT[:,j])
            bias_now = tot_rician_full_fit[j]
            amp_corr_now = rk.koay_basser_correction(col_now, bias_now, lambda_thresh=0.8)
            amp_corr[:,j] = amp_corr_now
            if j%100==0:
                print(j)
        
        phase = np.angle(self.signal_sb_FT)
        self.signal_sb_FT = amp_corr * np.exp(1j*phase)
        
        rk.plot_mat(self.signal_sb_FT+1e-20, extent=[self.E_lo,self.E_hi,hbar*self.OM_T[0,0],hbar*self.OM_T[-1,0]],
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
        
        amp_corr = np.zeros_like(self.sig_sb_FT_full_detrended)
        
        for j in range(self.N_E):
            col_now = np.abs(self.sig_sb_FT_full_detrended[:,j])
            bias_now = tot_rician_fit[j]
            amp_corr_now = rk.koay_basser_correction(col_now, bias_now)
            amp_corr[:,j] = amp_corr_now
            if j%100==0:
                print(j)
        
        phase = np.angle(self.sig_sb_FT_full_detrended)
        self.sig_sb_FT_full_detrended = amp_corr * np.exp(1j*phase)
        
        rk.plot_mat(self.sig_sb_FT_full_detrended, extent=[self.E_lo,self.E_hi,hbar*self.OM_T[0,0],hbar*self.OM_T[-1,0]],
                saveloc='single_output_temp/pipeline_diag/signal_FT_detrended_Koay-Basser.png', show=False)
        
        self.sig_probe_reconstructed,_,_,_ = rk.CFT(self.T_range, self.sig_sb_FT_full_detrended, use_window=False, inverse=True)
        
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
        xuv_sigq_avg = np.abs(xuv_sigq_avg).astype(float)
        xuv_sigq_avg *= np.max(np.abs(self.sp_xuv)**2)/np.max(xuv_sigq_avg)


        #### FIT

        # Fit two Gaussians with same sigma, separated by 0.177 eV
        # Model: A1/(2s)*exp(-0.5*((om-mu)/s)^2) + A2/(2s)*exp(-0.5*((om-(mu+0.177/hbar))/s)^2)
        # Free params: A1, A2, mu (center of first peak), s (shared width)
        om_axis = self.E[0,:] / hbar
        delta_om = 0.177 / hbar  # fixed splitting in angular-frequency units

        def _two_gauss_linked(om, A1, A2, mu, s):
            g1 = A1 / (2*s) * np.exp(-0.5 * ((om - mu) / s)**2)
            g2 = A2 / (2*s) * np.exp(-0.5 * ((om - (mu + delta_om)) / s)**2)
            return g1 + g2

        # Initial guesses
        om_max = om_axis[np.argmax(xuv_sigq_avg)]
        s0 = 0.08 / hbar
        A0 = float(np.max(xuv_sigq_avg)) * 2 * s0

        from scipy.optimize import curve_fit as _cf
        try:
            popt, _ = _cf(_two_gauss_linked, om_axis, xuv_sigq_avg,
                          p0=[A0, A0, om_max - delta_om, s0],
                          bounds=([0, 0, om_axis[0], 1e-12],
                                  [np.inf, np.inf, om_axis[-1], (om_axis[-1]-om_axis[0])]),
                          maxfev=20000)
            A1_fit, A2_fit, mu_fit, s_fit = popt
        except Exception:
            A1_fit, A2_fit, mu_fit, s_fit = A0, A0, om_max - delta_om, s0

        # Pack into the same [A1, mu1, s1, A2, mu2, s2] format
        sp_xuv_meas_sig_fit_params = np.array([A1_fit, mu_fit, s_fit,
                                                A2_fit, mu_fit + delta_om, s_fit])
        self.sp_xuv_meas_sig_fit = rk._sum_n_gauss1d(om_axis, *sp_xuv_meas_sig_fit_params)

        #### END FIT
        
        self.xuvs_rec = rk.nfit_params_to_probes(sp_xuv_meas_sig_fit_params, self.T)
        self.sp_xuv_meas_sig_fit = rk.sp_tot(self.xuvs_rec,self.om_xuv)
        
        diff = self.xuvs_rec[0][2]*hbar - self.xuvs_rec[1][2]*hbar
        ratio = self.xuvs_rec[0][1]/self.xuvs_rec[1][1]


        om_max = self.E[0,np.argmax(xuv_sigq_avg)]/hbar
        s_guess = 0.08/hbar
        a_guess = 0.01
        sp_ideal = rk.sp_tot(((0,a_guess*1,om_max-0.177/hbar,s_guess,0),(0,a_guess*2,om_max,s_guess,0)),self.E[0,:]/hbar)

        # # --- Richardson-Lucy deconvolution: find positive kernel h such that sp_ideal ∗ h ≈ xuv_sigq_avg ---
        # # Normalize both signals so the convolution is well-conditioned
        # obs = np.maximum(xuv_sigq_avg, 0.0)
        # psf = np.maximum(sp_ideal, 0.0)
        # obs_sum = obs.sum()
        # psf_sum = psf.sum()
        # if obs_sum > 0 and psf_sum > 0:
        #     obs_n = obs / obs_sum
        #     psf_n = psf / psf_sum
        #     # Initialize kernel as flat
        #     n_pts = obs_n.size
        #     h_rl = np.ones(n_pts) / n_pts
        #     n_rl_iter = 3000
        #     eps_rl = 1e-12
        #     for _rl_i in range(n_rl_iter):
        #         conv_fwd = fftconvolve(psf_n, h_rl, mode='same')
        #         ratio_rl = obs_n / (conv_fwd + eps_rl)
        #         correction = fftconvolve(ratio_rl, psf_n[::-1], mode='same')
        #         h_rl = h_rl * correction
        #         h_rl = np.maximum(h_rl, 0.0)
        #     # Reconvolve for comparison
        #     reconvolved_n = fftconvolve(psf_n, h_rl, mode='same')
        #     # Scale back to original units
        #     reconvolved = reconvolved_n * obs_sum
        #     h_kernel = h_rl  # keep normalized kernel
        # else:
        #     h_kernel = np.zeros_like(obs)
        #     reconvolved = np.zeros_like(obs)

        # # Save kernel to file so it can be loaded for experimental-signal deconvolution
        # # np.save('single_output_temp/reconstructions/h_kernel.npy', h_kernel)

        # E_axis = self.E[0, :]
        # dE = E_axis[1] - E_axis[0]
        # # Kernel axis centered at zero
        # h_axis = np.arange(n_pts) * dE
        # h_axis = h_axis - h_axis[n_pts // 2]

        # # Plot 1: broadening kernel h
        # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # axes[0].plot(h_axis, h_kernel, color='teal', linewidth=1.5)
        # axes[0].set_xlabel('Energy shift [eV]')
        # axes[0].set_ylabel('Amplitude')
        # axes[0].set_title('Deconvolved broadening kernel h')
        # axes[0].set_xlim([-1.0, 1.0])
        # axes[0].grid(True, alpha=0.3)

        # # Plot 2: reconvolution comparison
        # axes[1].plot(E_axis, xuv_sigq_avg, label='measured xuv band', linewidth=1.5)
        # axes[1].plot(E_axis, reconvolved, '--', label='sp_ideal ∗ h (reconvolved)', linewidth=1.5)
        # axes[1].plot(E_axis, sp_ideal * obs_sum / psf_sum if psf_sum > 0 else sp_ideal,
        #              ':', label='model sp_ideal (scaled)', linewidth=1.0, alpha=0.6)
        # axes[1].set_xlim([self.harmq_lo - 0.2, self.harmq_hi + 0.2])
        # axes[1].set_xlabel('Energy [eV]')
        # axes[1].set_ylabel('Amplitude')
        # axes[1].set_title('Reconvolution check')
        # axes[1].legend()
        # axes[1].grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.savefig('single_output_temp/reconstructions/xuv_deconvolution.png', dpi=300)
        # plt.close()

        plt.plot(self.E[0,:],xuv_sigq_avg,label='measured xuv band')
        plt.plot(self.E[0,:],sp_ideal,label='model xuv spectrum')
        plt.plot(self.om_xuv*hbar,self.sp_xuv_meas_sig_fit,linewidth=0.65,linestyle='--',label='reconstructed xuv sp.')
        plt.xlim([self.harmq_lo-0.2,self.harmq_hi+0.2])
        plt.xlabel('Energy [eV]')
        plt.ylabel('Amplitude (normalized)')
        plt.title('XUV spectrum rec, gaussian fit')
        plt.legend()
        plt.text(0.2, 0.9, f'split: {diff:.03f} eV\nratio:{ratio:.02f}', transform=plt.gca().transAxes, ha='center', va='bottom')
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
                                             lastmax_margin=np.sqrt(self.alpha)*700*np.sqrt(self.N_T/350), eps=self.WF_eps,
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
        self.amplit_tot_FT_corrected = correction*self.signal_sb_FT
        self.amplit_tot_FT_corrected = median_filter(np.abs(self.amplit_tot_FT_corrected), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected))
        
        correction_x = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs_rec, self.om_xuv), 
                                             rk.normalize_params(self.probes, self.om_probe), 
                                             dzeta=self.dzeta_val, theta=self.theta_val)
        self.amplit_tot_FT_corrected_x = correction_x*self.signal_sb_FT
        self.amplit_tot_FT_corrected_x = median_filter(np.abs(self.amplit_tot_FT_corrected_x), size=(3,3))*np.exp(1j*np.angle(self.amplit_tot_FT_corrected_x))
        
        correction_rec_x = rk.correcting_function_multi(self.OM_T, self.E, rk.normalize_params(self.xuvs_rec, self.om_xuv), 
                                             rk.normalize_params(self.probes_reconstructed, self.om_probe), 
                                             dzeta=self.dzeta_val, theta=self.theta_val)
        self.amplit_tot_FT_corrected_rec_x_nomedian = correction_rec_x*self.signal_sb_FT
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
        # self.amplit_tot_FT_corrected = correction * self.signal_sb_FT
        # self.amplit_tot_FT_corrected = median_filter(np.abs(self.amplit_tot_FT_corrected), size=(3,3)) * np.exp(1j*np.angle(self.amplit_tot_FT_corrected))

        # # --- Correction 2: reconstructed XUV + true probe ---
        # sp_xuv_rec = rk.sp_tot(self.xuvs_rec, self.om_xuv_up)
        # correction_x = rk.correcting_function_synth(
        #     self.T_range, self.T_up,
        #     sp_probe_corr, sp_xuv_rec, self.om_probe_up, self.om_xuv_up,
        #     sp_ref=sp_ref_corr, p_E=self.p_E,
        #     dzeta=self.dzeta_val, theta=self.theta_val)
        # self.amplit_tot_FT_corrected_x = correction_x * self.signal_sb_FT
        # self.amplit_tot_FT_corrected_x = median_filter(np.abs(self.amplit_tot_FT_corrected_x), size=(3,3)) * np.exp(1j*np.angle(self.amplit_tot_FT_corrected_x))

        # # --- Correction 3: reconstructed XUV + reconstructed probe ---
        # sp_probe_rec = rk.sp_tot(self.probes_reconstructed, self.om_probe_up)
        # correction_rec_x = rk.correcting_function_synth(
        #     self.T_range, self.T_up,
        #     sp_probe_rec, sp_xuv_rec, self.om_probe_up, self.om_xuv_up,
        #     sp_ref=sp_ref_corr, p_E=self.p_E,
        #     dzeta=self.dzeta_val, theta=self.theta_val)
        # self.amplit_tot_FT_corrected_rec_x_nomedian = correction_rec_x * self.signal_sb_FT
        # self.amplit_tot_FT_corrected_rec_x = median_filter(np.abs(self.amplit_tot_FT_corrected_rec_x_nomedian), size=(3,3)) * np.exp(1j*np.angle(self.amplit_tot_FT_corrected_rec_x_nomedian))
    
    def resample_analyze(self):
        """Resample and analyze results to produce final density matrices."""
        # Resample and analyze

        rho_uncorrected, amplit_tot_FT_corrected_small, extent_small, idxs_small, _, _ = rk.resample(
            median_filter(np.abs(self.signal_sb_FT),size=(3,3))*np.exp(1j*np.angle(self.signal_sb_FT)), self.rho_hi, self.rho_lo, self.om0_ref, self.E, self.OM_T, self.N_T)
        
        rk.plot_mat(rho_uncorrected, extent=[self.rho_lo,self.rho_hi,self.rho_lo,self.rho_hi], cmap='plasma',
                 mode='abs', saveloc='single_output_temp/rhos/synth_uncorr_unproj.png', xlabel='Energy [eV]', ylabel='Energy [eV]',
                 title='Rho uncorrected for the probe spectrum', show=False)
        
        rho_uncorrected = rk.project_to_density_matrix(rho_uncorrected)

        rho_reconstructed, amplit_tot_FT_corrected_small, extent_small, idxs_small, E1, E2 = rk.resample(
            self.amplit_tot_FT_corrected, self.rho_hi, self.rho_lo, self.om0_ref, self.E, self.OM_T, self.N_T)
        rho_reconstructed = rk.project_to_density_matrix(rho_reconstructed)
        
        rho_reconstructed_x, amplit_tot_FT_corrected_small, extent_small, idxs_small, _, _ = rk.resample(
            self.amplit_tot_FT_corrected_x, self.rho_hi, self.rho_lo, self.om0_ref, self.E, self.OM_T, self.N_T)
        rho_reconstructed_x = rk.project_to_density_matrix(rho_reconstructed_x)
        
        rho_reconstructed_rec_x, amplit_tot_FT_corrected_small_rec, extent_small, idxs_small, _, _ = rk.resample(
            self.amplit_tot_FT_corrected_rec_x, self.rho_hi, self.rho_lo, self.om0_ref, self.E, self.OM_T, self.N_T)
        rho_reconstructed_rec_x = rk.project_to_density_matrix(rho_reconstructed_rec_x)
        
        rho_reconstructed_rec_x_nomedian, amplit_tot_FT_corrected_small_rec, extent_small, idxs_small, _, _ = rk.resample(
            self.amplit_tot_FT_corrected_rec_x_nomedian, self.rho_hi, self.rho_lo, self.om0_ref, self.E, self.OM_T, self.N_T)
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

            E_lo = 21.0
            # E_hi = 29
            E_hi = 29.0
            T_reach = 50
            # T_reach = 250
            E_res = 0.025    
            # E_res = 0.005    
            N_T = 260
            # N_T = 700
            p_E = 4  # N_E upsampling integer
            alpha = 1000000
            # alpha = 10000
            b = 1

            sideband_lo = 26.0
            sideband_hi = 27.5
            harmq_lo = 24.0
            harmq_hi = 26.0

            if_coherent = True

            # Create experiment instance
            experiment = RK_experiment(E_lo=E_lo,E_hi=E_hi,T_reach=T_reach,E_res=E_res,N_T=N_T,p_E=p_E,alpha=alpha,b=b,
                                    sb_lo=sideband_lo,sb_hi=sideband_hi,harmq_lo=harmq_lo,harmq_hi=harmq_hi,if_coherent=if_coherent)
            
            # Define pulses
            # A_xuv = 1.0
            # a_xuvs = [1.0,1.0]
            # om_xuvs = [10.65/hbar,(10.65+2*1.55)/hbar]
            # s_xuvs = [0.15/hbar,0.18/hbar]

            # A_xuv = 0.1
            # a_xuvs = [1.0]
            # om_xuvs = [(25.65-1*1.40)/hbar]
            # s_xuvs = [0.15/hbar]

            # A_xuv = 0.1
            # a_xuvs = [np.sqrt(0.4),np.sqrt(0.8)]
            # om_xuvs = [(25.65-1*1.75)/hbar,(25.65-1*1.75+0.27)/hbar]
            # s_xuvs = [0.10/hbar,0.10/hbar]

            scale = 0.2

            A_xuv = 0.1*scale
            a_xuvs = [np.sqrt(0.8),np.sqrt(0.8),np.sqrt(0.8),np.sqrt(0.8),np.sqrt(0.8),np.sqrt(0.8)]
            om_xuvs = [(25.65-1*1.75)/hbar,(25.65+1*1.45)/hbar,(25.65+1*1.45+3.10)/hbar,(25.65-1*1.45)/hbar,(25.65+1*1.75)/hbar,(25.65+1*1.75+3.10)/hbar]
            s_xuvs = [0.1/hbar,0.1/hbar,0.1/hbar,0.1/hbar,0.1/hbar,0.1/hbar]

            a_xuvs = [np.sqrt(0.8),np.sqrt(0.8),np.sqrt(0.8),
                      np.sqrt(0.8),np.sqrt(0.8),np.sqrt(0.8)]
            om_xuvs = [(25.00-0.2)/hbar,(25.00+3.1-0.2)/hbar,(25.00-3.1-0.2)/hbar,
                       (25.00+0.2)/hbar,(25.00+3.1+0.2)/hbar,(25.00-3.1+0.2)/hbar]
            s_xuvs = [0.1/hbar,0.1/hbar,0.1/hbar,
                      0.1/hbar,0.1/hbar,0.1/hbar]
            
            a_xuvs = [0.8,0.4]
            om_xuvs = [(25.00-0.2)/hbar,
                       (25.00+0.2)/hbar]
            s_xuvs = [0.1/hbar,0.1/hbar]


            A_probe = 1.2*scale
            a_probes = [1.0,0.2,0.2,0.3]
            om_probes = [1.55/hbar,1.20/hbar,2.00/hbar,1.85/hbar]
            s_probes = [0.15/hbar,0.04/hbar,0.07/hbar,0.17/hbar]

            A_probe = 1.2*scale
            a_probes = [1.0,0.2]
            om_probes = [1.55/hbar,1.75/hbar]
            s_probes = [0.11/hbar,0.05/hbar]

            # # Flat-top probe (similar total width and avg amplitude)
            # A_probe = 1.2
            # a_probes = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35]
            # om_probes = [1.25/hbar, 1.39/hbar, 1.53/hbar, 1.67/hbar, 1.81/hbar, 1.95/hbar, 2.09/hbar]
            # s_probes = [0.10/hbar, 0.10/hbar, 0.10/hbar, 0.10/hbar, 0.10/hbar, 0.10/hbar, 0.10/hbar]

            A_ref = 0.6*scale
            a_refs = [1.0,0.001]
            om_refs = [1.25/hbar,1.55/hbar]
            s_refs = [0.05/hbar,0.15/hbar]

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