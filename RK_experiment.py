"""
Rainbow-KRAKEN WF Pipeline - Refactored Class-based Implementation

This module contains the RK_experiment class which encapsulates the full 
Rainbow-KRAKEN signal processing pipeline for quantum state reconstruction.

The pipeline consists of the following stages:
1. prepare_grid() - Initialize energy/time grids and pulse configurations
2. generate_signal() - Generate synthetic baseline and full signal with noise
3. process_and_detrend() - Process signal, estimate background, and detrend
4. kb_correct() - Apply Koay-Basser correction for Rician noise
5. probe_reconstruct() - Reconstruct probe spectrum from sideband FT
6. probe_sp_correct() - Apply probe correction using true and reconstructed probes
7. mcmc_fit() - Fit the corrected density matrix with Bayesian MCMC

Usage:
    experiment = RK_experiment()
    experiment.generate_signal()
    experiment.process_and_detrend() 
    experiment.kb_correct()
    experiment.probe_reconstruct()
    experiment.probe_sp_correct()
    experiment.mcmc_fit()

All experimental parameters are stored as class attributes and can be 
modified before running the pipeline methods.
"""

import numpy as np
import gc
from numpy import pi
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import median_filter
import rkraken as rk
from matplotlib import patheffects as pe
from LBFGSprobe import LBFGS_probe
from MCMCrho import Bayesian_MCMC

# Reduced Planck constant in eV*fs (approx CODATA): ħ = 6.582119569e-16 eV·s
hbar = 6.582119569e-1

class RK_experiment:
    """Rainbow-KRAKEN experiment class for full pipeline processing."""
    
    def __init__(self,E_lo=60.0,E_hi=63.5,T_reach=50,E_res=0.025,N_T=240,alpha=0.05,b=1,sb_lo=24.7,sb_hi=28,harmq_lo=22,harmq_hi=24.7,
                 A_ref=1,om_ref=1.55/hbar,s_ref=0.02/hbar):

        ###
        ### KEYWORD PARAMETERS
        ###

        # Field parameters
        self.E_lo = E_lo
        self.E_hi = E_hi
        self.T_reach = T_reach

        self.E_res = E_res
        self.N_E = round((E_hi - E_lo)/self.E_res/10)*10
        self.N_T = N_T
        
        self.alpha = alpha
        self.b = b

       # filter parameters
        self.A_ref = A_ref
        self.om_ref = om_ref
        self.s_ref = s_ref
        
        # Band parameters
        self.sb_lo = sb_lo
        self.sb_hi = sb_hi
        self.harmq_lo = harmq_lo # Sets interpolation region!
        self.harmq_hi = harmq_hi # Sets interpolation region!

        ###
        ### DEFAULT PARAMETERS
        ###
        
        # Noise and processing parameters
        self.zero_pad = 0
        self.ifnoise = True

        # pipeline modes
        self.ifWide = False
        self.median_filter_when = 1 if self.ifWide else 0 # 0 - never, 1 - after kbcorr, 2 - after spcorr

        # KB correction parameters
        self.lambda_thresh = 1.0
        self.y_lo = 1.0/hbar # This is used just for plotting - ROI selection
        self.y_hi = 2.0/hbar

        # Probe reconstruction parameters
        self.prrec_dzeta = 0.02
        self.prrec_maxiter = 500
        self.prrec_tol = 1e-6
        self.prrec_lambda1 = 0.0
        self.prrec_lambda2 = 0.005

        # Probe correction parameters / MCMC data region parameters
        self.prcor_dzeta = 0.35 if self.ifWide else 0.25
        self.rho_lo = 24.0 + om_ref*hbar
        self.rho_hi = 26.0 + om_ref*hbar

        # Rho retrieval parameters
        self.N_NEW = 100
        self.mcmc_peaks = 2
        self.psd_sigma = 4
                
        # MCMC data region parameters
        self.mcmc_num_warmup = 1000
        self.mcmc_num_samples = 2000
        self.mcmc_num_chains = 1
        
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

    def define_pulses(self, probe_params):
        """Define probe pulse configurations using a parameter dictionary."""
        self.probe_params = probe_params

    def sp_tot(self, om):
        om = np.asarray(om)
        retval = np.zeros_like(om, dtype=np.complex128)

        amps = self.probe_params['amps']
        oms = self.probe_params['oms']
        sigmas = self.probe_params['sigmas']
        phi0 = self.probe_params.get('phi0', 0.0)
        phase_grad = self.probe_params.get('phase_grad', 0.0)
        phase_chirp = self.probe_params.get('phase_chirp', 0.0)

        n_peaks = len(amps)
        if n_peaks == 0:
            return retval

        om_ref = oms[np.argmax(np.abs(amps))]
        domega = om - om_ref

        phase = phi0 + phase_grad * domega + 0.5 * phase_chirp * domega**2

        for i in range(n_peaks):
            retval += rk.spectrum_fun(amps[i], oms[i], sigmas[i], om) * np.exp(1j * phase)
            
        return retval

    def define_model(self, rho_params):
        # Keep known rho parameters for probe-only Bayesian inference.
        self.rho_params = rho_params

        amps = rho_params['amps']
        mus = rho_params['mus']
        sigmas = rho_params['sigmas']
        betas = rho_params['betas']
        taus = rho_params['taus']
        lambdas = rho_params['lambdas']
        gammas = rho_params['gammas']
        etas = rho_params['etas']

        def rho_f(e1, e2):
            return rk.rho_model(e1,e2,amps,mus,sigmas,betas,taus,lambdas,gammas,etas) 

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

        ### PLOT the upsampled spectra with probe/ref on one axis, xuv on another

        np.savez('single_output_temp/1generate_signal/input_spectra.npz',
                 om_probe=self.om_probe,om_xuv=self.om_xuv,sp_probe=self.sp_probe,sp_ref=self.sp_ref,
                 sp_xuv=np.abs(self.rho_f(self.om_xuv*hbar + self.om_ref*hbar,self.om_xuv*hbar + self.om_ref*hbar)))

        # rk.plot_spectra(self.om_probe,self.om_xuv,self.sp_probe,self.sp_ref,np.abs(self.rho_f(self.om_xuv*hbar + self.om_ref*hbar,self.om_xuv*hbar + self.om_ref*hbar)))
        
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

        np.savez('single_output_temp/1generate_signal/exact_freqsig.npz',
                 mat_complex=self.signal_ft0,extent=np.array([self.E_lo,self.E_hi,em_lo,em_hi]))
        np.savez('single_output_temp/1generate_signal/exact_timesig.npz',
                 mat_abs=signal_clean,extent=np.array([self.E_lo,self.E_hi,-self.T_reach,self.T_reach]))

        # rk.plot_mat(self.signal_ft0-1e-4,extent=[self.E_lo,self.E_hi,em_lo,em_hi],saveloc='single_output_temp/pipeline_diag/model_ft.png')
        # rk.plot_mat(signal_clean-1e-4,extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],saveloc='single_output_temp/pipeline_diag/model_sig.png')

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

        # rk.plot_mat(self.signal, extent=[self.E_lo,self.E_hi,-self.T_reach,self.T_reach],
        #         caption='Peak SB counts: %i\nN_T: %i\nN_E: %i\nT_res: %.2f fs\nE_res: %.3f eV'%(
        #             self.peak_sb_counts,self.N_T,self.N_E,2*self.T_reach/self.N_T,self.E_res),
        #         saveloc='single_output_temp/pipeline_diag/measured_signal.png',xlabel='Kinetic energy $E_f$ (eV)',ylabel='Time delay $\\tau$ (fs)',mode='abs',title='Simulated noisy signal')

    def process_and_detrend(self):
        
        self.signal_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal, use_window=False, zero_pad=self.zero_pad)
        self.signal_sb_FT, self.OM_T, em_lo, em_hi = rk.CFT(self.T_range, self.signal_sb, use_window=False, zero_pad=self.zero_pad)

        # rk.plot_mat(self.signal_FT, extent=[self.E_lo,self.E_hi,em_lo,em_hi],
        #         saveloc='single_output_temp/pipeline_diag/signal_FT.png', show=False)

        np.savez('single_output_temp/2process_detrend/measured_freqsig.npz',
                 mat_complex=self.signal_FT,extent=np.array([self.E_lo,self.E_hi,em_lo,em_hi]))
        
        ### ROI SLICE FRACTS

        Ef_reach = self.E[0,-1] - self.E[0,1]
        self.Ef_slice_fracts = ((self.rho_lo - self.E[0,0])/Ef_reach, (self.rho_hi - self.E[0,0])/Ef_reach)

        OMt_reach = self.OM_T[-1,0] - self.OM_T[1,0]
        self.OMt_slice_fracts = ((self.y_lo - self.OM_T[1,0])/OMt_reach, (self.y_hi - self.OM_T[1,0])/OMt_reach)

        self.signal_ft0_ROI, em_axis_mid, i0, i1, ef_axis_mid, e0, e1 = rk.extract_midslice(self.signal_ft0, self.OMt_slice_fracts, hbar*self.OM_T[:,0], self.Ef_slice_fracts, self.E)
        signal_sb_FT_ROI, em_axis_mid, i0, i1, ef_axis_mid, e0, e1 = rk.extract_midslice(self.signal_sb_FT, self.OMt_slice_fracts, hbar*self.OM_T[:,0], self.Ef_slice_fracts, self.E)
        
        # rk.plot_mat(signal_sb_FT_ROI, extent=[self.rho_lo,self.rho_hi,self.y_lo*hbar,self.y_hi*hbar],
        #         saveloc='single_output_temp/pipeline_diag/signal_FT_sb_ROI.png', xlabel='Kinetic energy $E_f$ (eV)',ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)', show=False,title='Sideband region of interest before KB correction', 
        #         caption=f'RES = {np.sum(np.abs(self.signal_ft0_ROI - signal_sb_FT_ROI)) / np.sum(np.abs(self.signal_ft0_ROI)):.4f}')
        np.savez('single_output_temp/3kb_correct/KB_before.npz',
                 mat_complex=signal_sb_FT_ROI,extent=np.array([self.rho_lo,self.rho_hi,self.y_lo*hbar,self.y_hi*hbar]),
                 RES=np.sum(np.abs(self.signal_ft0_ROI - signal_sb_FT_ROI)**2) / np.sum(np.abs(self.signal_ft0_ROI)**2))
        
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
            amp_corr[:,j] = rk.koay_basser_correction(col_now, bias_now, lambda_thresh=self.lambda_thresh, only_floor = False)
        phase = np.angle(self.signal_sb_FT)

        self.signal_sb_FT = amp_corr * np.exp(1j*phase)

        signal_sb_FT_ROI, em_axis_mid, i0, i1, ef_axis_mid, e0, e1 = rk.extract_midslice(self.signal_sb_FT, self.OMt_slice_fracts, hbar*self.OM_T[:,0], self.Ef_slice_fracts, self.E)
        # rk.plot_mat(signal_sb_FT_ROI - 1e-5, extent=[self.rho_lo,self.rho_hi,self.y_lo*hbar,self.y_hi*hbar],
        #         saveloc='single_output_temp/pipeline_diag/signal_FT_sb_KB_corr.png', xlabel='Kinetic energy $E_f$ (eV)',ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)', show=False,title='Sideband region of interest after KB correction', 
        #         caption=f'RES = {np.sum(np.abs(self.signal_ft0_ROI - signal_sb_FT_ROI)) / np.sum(np.abs(self.signal_ft0_ROI)):.4f}')
        np.savez('single_output_temp/3kb_correct/KB_after.npz',
                 mat_complex=signal_sb_FT_ROI,extent=np.array([self.rho_lo,self.rho_hi,self.y_lo*hbar,self.y_hi*hbar]),
                 RES=np.sum(np.abs(self.signal_ft0_ROI - signal_sb_FT_ROI)**2) / np.sum(np.abs(self.signal_ft0_ROI)**2))
        
        if self.median_filter_when == 1:
            self.signal_sb_FT = median_filter(np.real(self.signal_sb_FT),size=(3,3)) + 1j*median_filter(np.imag(self.signal_sb_FT),size=(3,3))

        # rk.plot_mat(np.abs(self.signal_ft0_ROI -  signal_sb_FT_ROI ) / np.max(np.abs(self.signal_ft0_ROI)), extent=[self.E_lo,self.E_hi,hbar*self.OM_T[0,0],hbar*self.OM_T[-1,0]],
        #         saveloc='single_output_temp/pipeline_diag/signal_FT_sb_KB_corr_diff.png', xlabel='Kinetic energy $E_f$ (eV)',ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)', show=False,title='FT of the sideband signal', 
        #         caption=f'RES = {np.sum(np.abs(self.signal_ft0_ROI - signal_sb_FT_ROI)) / np.sum(np.abs(self.signal_ft0_ROI)):.4f}')

        W = np.repeat((np.abs(self.sp_tot(em_axis_mid/hbar) / em_axis_mid)**2)[:,np.newaxis], signal_sb_FT_ROI.shape[1], axis=1)
        P_SIG = median_filter(np.abs(signal_sb_FT_ROI)**2,size=(4,4))
        P_NOISE = (self.sigma[i0:i1,e0:e1])**2
        P_NOISE[P_SIG==0] = 0

        self.P_SNR = np.sum(P_SIG*W)/np.sum(P_NOISE*W)

        np.savez('single_output_temp/1generate_signal/measured_timesig.npz',
                 mat_abs=self.signal,extent=np.array([self.E_lo,self.E_hi,-self.T_reach,self.T_reach]),
                 P_SNR=self.P_SNR,N_T=self.N_T,N_E=self.N_E,T_res=2*self.T_reach/self.N_T,E_res=self.E_res)


    def probe_reconstruct(self):

        sig_power = np.sum(np.abs(self.signal_sb_FT),axis=1)

        x_mask = (self.E[0, :] > self.rho_lo) & (self.E[0, :] < self.rho_hi)
        y_mask =  (sig_power > self.prrec_dzeta * np.max(sig_power)) & (np.abs(self.OM_T)[:,0] < self.om_ref/2) & (self.OM_T[:,0] > np.min(np.abs(self.OM_T[:,0])))

        self.E_zero = self.E[np.ix_(y_mask, x_mask)]
        self.OM_T_zero = self.OM_T[np.ix_(y_mask, x_mask)]
        self.sigma_zero = self.sigma[np.ix_(y_mask, x_mask)]
        self.sigma_zero = np.abs(self.sigma_zero).astype(float)
        self.signal_sb_FT_zero = self.signal_sb_FT[np.ix_(y_mask, x_mask)]

        # rk.plot_mat(self.signal_sb_FT_zero - 1e-3, extent=[self.E_zero[0,0]-self.om_ref*hbar,self.E_zero[0,-1]-self.om_ref*hbar,self.OM_T_zero[1,0]*hbar,self.OM_T_zero[-1,0]*hbar], cmap='plasma',
        #          saveloc='single_output_temp/LBFGS/zero_comp.png', xlabel='Kinetic energy $E_f$ (eV)', ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
        #          title='$\\tilde S_{corr}(E_f,\\omega_\\tau)$', show=False, square=True)
        np.savez('single_output_temp/3kb_correct/zero_omega_comp.npz',
                 mat_complex=self.signal_sb_FT_zero,extent=np.array([self.E_zero[0,0]-self.om_ref*hbar,self.E_zero[0,-1]-self.om_ref*hbar,self.OM_T_zero[1,0]*hbar,self.OM_T_zero[-1,0]*hbar]))
        
        # rk.plot_mat(self.sigma_zero, extent=[self.E_zero[0,0]-self.om_ref*hbar,self.E_zero[0,-1]-self.om_ref*hbar,self.OM_T_zero[1,0]*hbar,self.OM_T_zero[-1,0]*hbar], cmap='plasma',
        #          saveloc='single_output_temp/LBFGS/zero_comp_sigma.png', xlabel='Kinetic energy $E_f$ (eV)', ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
        #          title='$\\sigma(E_f,\\hbar \\omega_\\tau )$', show=False, mode='abs')
        np.savez('single_output_temp/3kb_correct/zero_omega_comp_sigma.npz',
                 mat_abs=self.sigma_zero,extent=np.array([self.E_zero[0,0]-self.om_ref*hbar,self.E_zero[0,-1]-self.om_ref*hbar,self.OM_T_zero[1,0]*hbar,self.OM_T_zero[-1,0]*hbar]))
        
        ir_lo,xuv_lo,ir_hi,xuv_hi = rk.conv_bounds(self.rho_lo/hbar,self.rho_hi/hbar,self.N_E,self.om_ref)
        om_probe_zero = np.linspace(ir_lo,ir_hi,self.E_zero.shape[1])
        om_xuv_zero = np.linspace(xuv_lo,xuv_hi,self.E_zero.shape[1])
        OM_P_zero = np.tile(om_probe_zero, (self.E_zero.shape[0], 1))
        OM_X_zero = np.tile(om_xuv_zero, (self.E_zero.shape[0], 1))

        # # Build full frequency grids used by the zero-omega forward model and apply the same ROI masks.
        # OM_P = np.tile(self.om_probe, (self.N_T, 1))
        # OM_X = np.tile(self.om_xuv, (self.N_T, 1))
        # OM_P_zero = OM_P[np.ix_(y_mask, x_mask)]
        # OM_X_zero = OM_X[np.ix_(y_mask, x_mask)]

        opt_probe, z_fit_zero = LBFGS_probe(
            om_p=OM_P_zero,
            om_x=OM_X_zero,
            om_t=self.OM_T_zero,
            z_obs=self.signal_sb_FT_zero,
            sigma_obs=self.sigma_zero,
            rho_params=self.rho_params,
            om_ref=self.om_ref,
            obs_mask=None,
            maxiter=self.prrec_maxiter,
            tol=self.prrec_tol,
            lambda1=self.prrec_lambda1,
            lambda2=self.prrec_lambda2
        ) # z_fit_zero: Forward model from reconstructed probe

        # Interpolate the optimized discrete vector to the full probe frequency grid if necessary
        opt_probe_interp = (np.interp(self.om_probe, om_probe_zero, np.real(opt_probe),left=0.0,right=0.0) 
                            + 1j * np.interp(self.om_probe, om_probe_zero, np.imag(opt_probe),left=0.0,right=0.0))

        self.sp_probe_inferred_complex = opt_probe_interp
        
        print("Inferred probe field from discrete L-BFGS completed.")

        true_power = np.sum(np.abs(self.sp_probe)**2)
        fit_power = np.sum(np.abs(self.sp_probe_inferred_complex)**2)
        fit_mag = np.abs(self.sp_probe_inferred_complex * np.sqrt(true_power/fit_power))

        idx_max = np.argmax(np.abs(self.sp_probe))
        phase_offset = np.angle(self.sp_probe[idx_max]) - np.angle(self.sp_probe_inferred_complex[idx_max])
        fit_phase = np.angle(self.sp_probe_inferred_complex * np.exp(1j * phase_offset))

        res_val = np.sum(np.abs(self.sp_probe - fit_mag*np.exp(1j*fit_phase))**2) / np.sum(np.abs(self.sp_probe)**2)

        np.savez('single_output_temp/4probe_rec/probe_sp_rec.npz',
                 om_probe=self.om_probe,sp_probe=self.sp_probe,sp_probe_rec=fit_mag*np.exp(1j*fit_phase),
                 RES=res_val)


        # fig, ax_amp = plt.subplots(1, 1, figsize=(8, 4.5))
        # ax_phase = ax_amp.twinx()

        # true_power = np.sum(np.abs(self.sp_probe)**2)
        # fit_power = np.sum(np.abs(self.sp_probe_inferred_complex)**2)

        # true_mag = np.abs(self.sp_probe)
        # fit_mag = np.abs(self.sp_probe_inferred_complex * np.sqrt(true_power/fit_power))
        
        # # Calculate the relative global phase difference using maximum amplitude point
        # idx_max = np.argmax(true_mag)
        # phase_offset = np.angle(self.sp_probe[idx_max]) - np.angle(self.sp_probe_inferred_complex[idx_max])

        # true_phase = np.angle(self.sp_probe)
        # fit_phase = np.angle(self.sp_probe_inferred_complex * np.exp(1j * phase_offset))
        # true_phase_mask = true_mag >= 0.05 * np.max(true_mag)
        # fit_phase_mask = fit_mag >= 0.05 * np.max(fit_mag)
        # true_phase_plot = np.where(true_phase_mask, true_phase, np.nan)
        # fit_phase_plot = np.where(fit_phase_mask, fit_phase, np.nan)

        # line_true_amp, = ax_amp.plot(self.om_probe*hbar, true_mag, label='True probe |mag|', linewidth=1.2)
        # line_fit_amp, = ax_amp.plot(self.om_probe*hbar, fit_mag, '--', label='Inferred probe |mag|', linewidth=1.2)
        # ax_amp.set_xlabel('Energy [eV]')
        # ax_amp.set_ylabel('Amplitude [arb. u.]')
        # ax_amp.set_title('Probe spectrum: true vs inferred')
        # ax_amp.grid(True, alpha=0.3)
        # ax_amp.set_xlim([0.7,2.5])

        # line_true_phase, = ax_phase.plot(self.om_probe*hbar, true_phase_plot, label='True phase', linewidth=0.9)
        # line_fit_phase, = ax_phase.plot(self.om_probe*hbar, fit_phase_plot, '--', label='Inferred phase', linewidth=0.9)
        # ax_phase.set_ylabel('Phase [rad]')
        # ax_phase.set_ylim([-np.pi, np.pi])
        # ax_phase.set_yticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])

        # lines = [line_true_amp, line_fit_amp, line_true_phase, line_fit_phase]
        # labels = [line.get_label() for line in lines]
        # ax_amp.legend(lines, labels, loc='best')

        # res_val = np.sum(np.abs(self.sp_probe - fit_mag*np.exp(1j*fit_phase))**2) / np.sum(np.abs(self.sp_probe)**2)
        # self.probe_reconstruction_res = float(res_val)

        # ax_amp.text(0.02, 0.98, f"RES = {res_val:.4f}", transform=ax_amp.transAxes,
        # fontsize=10, verticalalignment='top', horizontalalignment='left',
        # color='white', weight='bold',
        # path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        # plt.tight_layout(rect=[0, 0.03, 1, 1])
        # plt.savefig('single_output_temp/pipeline_diag/probe_spectrum_fit.png', dpi=300)
        # plt.close(fig)

    def _apply_correction(self,sp,dzeta):
        ref_phase = np.exp(-1j*np.angle(np.interp(self.om_ref,self.OM_T[:,0],sp[:,0])))

        om_t_epsilon = 0.01
        probe_modulation = ref_phase * sp / np.maximum(self.OM_T,om_t_epsilon)

        where_off = np.abs(probe_modulation) < dzeta * np.max(np.abs(probe_modulation))
        probe_modulation[where_off] = dzeta

        signal_sb_FT_corrected = self.signal_sb_FT / probe_modulation
        signal_sb_FT_corrected[where_off] = 0

        sigma = self.sigma / np.abs(probe_modulation)
        sigma[where_off] = 0

        x_mask = (self.E[0, :] > self.rho_lo) & (self.E[0, :] < self.rho_hi)
        y_mask = np.abs(probe_modulation[:,0]) > dzeta

        if self.median_filter_when == 2:
            signal_sb_FT_corrected = median_filter(np.real(signal_sb_FT_corrected),size=(3,3)) + 1j*median_filter(np.imag(signal_sb_FT_corrected),size=(3,3))  # WIDE PROBE VARIANT

        E_rho = self.E[np.ix_(y_mask, x_mask)]
        OM_T_rho = self.OM_T[np.ix_(y_mask, x_mask)]
        sigma_rho = sigma[np.ix_(y_mask, x_mask)]
        sigma_rho = np.abs(sigma_rho).astype(float)
        signal_sb_FT_corrected_rho = signal_sb_FT_corrected[np.ix_(y_mask, x_mask)]

        om_row_idx = np.argmin(np.abs(OM_T_rho[:, 0] - self.om_ref))
        dE = E_rho[0, 1] - E_rho[0, 0]
        roi_norm = np.sum(np.abs(signal_sb_FT_corrected_rho)[om_row_idx, :]) * dE

        signal_sb_FT_corrected_rho = signal_sb_FT_corrected_rho / roi_norm
        sigma_rho = sigma_rho / roi_norm

        return signal_sb_FT_corrected, signal_sb_FT_corrected_rho, sigma, sigma_rho, E_rho, OM_T_rho

    def probe_sp_correct(self):

        sp_true = self.sp_tot(self.OM_T)

        sp_rec_col = (1+0j) * np.interp(self.OM_T[:,0], self.om_probe, np.real(self.sp_probe_inferred_complex), left=0.0, right=0.0)
        sp_rec_col += 1j * np.interp(self.OM_T[:,0], self.om_probe, np.imag(self.sp_probe_inferred_complex), left=0.0, right=0.0)
        sp_rec = np.repeat(sp_rec_col[:, np.newaxis], self.OM_T.shape[1], axis=1)

        self.rhodata, self.rhodata_roi, self.rhosigma, self.rhosigma_roi, self.E_roi, self.OM_T_roi = self._apply_correction(sp_true,self.prcor_dzeta)
        extent = np.array([self.E_roi[0,0]-self.om_ref*hbar,self.E_roi[0,-1]-self.om_ref*hbar,self.OM_T_roi[1,0]*hbar,self.OM_T_roi[-1,0]*hbar])

        # rk.plot_mat(self.rhosigma_roi, extent=[self.E_roi[0,0]-self.om_ref*hbar,self.E_roi[0,-1]-self.om_ref*hbar,self.OM_T_roi[1,0]*hbar,self.OM_T_roi[-1,0]*hbar], cmap='plasma',
        #          saveloc='single_output_temp/rhos/sigmas.png', xlabel='Kinetic energy $E_f$ (eV)', ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
        #          title='$\\sigma(E_f,\\hbar \\omega_\\tau )$', show=False, mode='abs')
        np.savez('single_output_temp/5probe_corr/data_sigma.npz',
                 mat_abs=self.rhosigma_roi,extent=extent)
        
        # rk.plot_mat(self.rhodata_roi - 1e-3, extent=[self.E_roi[0,0]-self.om_ref*hbar,self.E_roi[0,-1]-self.om_ref*hbar,self.OM_T_roi[1,0]*hbar,self.OM_T_roi[-1,0]*hbar], cmap='plasma',
        #          saveloc='single_output_temp/rhos/rho_unproj.png', xlabel='Kinetic energy $E_f$ (eV)', ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
        #          title='$\\tilde S_{corr}(E_f,\\omega_\\tau)$', show=False, square=True)
        np.savez('single_output_temp/5probe_corr/data_rho.npz',
                 mat_complex=self.rhodata_roi,extent=extent)
        
        self.rhodata_rec, self.rhodata_roi_rec, self.rhosigma_rec, self.rhosigma_roi_rec, self.E_roi_rec, self.OM_T_roi_rec = self._apply_correction(sp_rec,self.prcor_dzeta)
        extent = np.array([self.E_roi_rec[0,0]-self.om_ref*hbar,self.E_roi_rec[0,-1]-self.om_ref*hbar,self.OM_T_roi_rec[1,0]*hbar,self.OM_T_roi_rec[-1,0]*hbar])

        # rk.plot_mat(self.rhosigma_roi_rec, extent=[self.E_roi_rec[0,0]-self.om_ref*hbar,self.E_roi_rec[0,-1]-self.om_ref*hbar,self.OM_T_roi_rec[1,0]*hbar,self.OM_T_roi_rec[-1,0]*hbar], cmap='plasma',
        #          saveloc='single_output_temp/rhos/sigmas_rec.png', xlabel='Kinetic energy $E_f$ (eV)', ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
        #          title='$\\sigma(E_f,\\hbar \\omega_\\tau )$', show=False, mode='abs')
        np.savez('single_output_temp/5probe_corr/data_sigma_rec.npz',
                 mat_abs=self.rhosigma_roi_rec,extent=extent)
        
        # rk.plot_mat(self.rhodata_roi_rec - 1e-3, extent=[self.E_roi_rec[0,0]-self.om_ref*hbar,self.E_roi_rec[0,-1]-self.om_ref*hbar,self.OM_T_roi_rec[1,0]*hbar,self.OM_T_roi_rec[-1,0]*hbar], cmap='plasma',
        #          saveloc='single_output_temp/rhos/rho_unproj_rec.png', xlabel='Kinetic energy $E_f$ (eV)', ylabel='Indirect energy $\\hbar \\omega_\\tau$ (eV)',
        #          title='$\\tilde S_{corr}(E_f,\\omega_\\tau)$', show=False, square=True)
        np.savez('single_output_temp/5probe_corr/data_rho_rec.npz',
                 mat_complex=self.rhodata_roi_rec,extent=extent)
        
        return
    
    def _apply_mcmc(self, rhodata, rhodata_roi, rhosigma, rhosigma_roi, E_roi, OM_T_roi, suffix):

        rho_raw, _, _, _, self.E1interp, self.E2interp = rk.resample(
            rhodata, self.harmq_hi, self.harmq_lo, self.om_ref, self.E, self.OM_T, self.N_NEW)
        rho_raw_sigma, _, _, _, _, _ = rk.resample(
            rhosigma, self.harmq_hi, self.harmq_lo, self.om_ref, self.E, self.OM_T, self.N_NEW)
        
        raw_trace = np.trace(rho_raw)
        rho_raw = rho_raw/raw_trace
        rho_raw_sigma = rho_raw_sigma/raw_trace

        rho_raw[(self.E1interp - self.E2interp) < -np.max(OM_T_roi-self.om_ref)*hbar] = 0
        rho_raw[(self.E1interp - self.E2interp) > -np.min(OM_T_roi-self.om_ref)*hbar] = 0

        rho_raw_sigma[(self.E1interp - self.E2interp) < -np.max(OM_T_roi-self.om_ref)*hbar] = 0
        rho_raw_sigma[(self.E1interp - self.E2interp) > -np.min(OM_T_roi-self.om_ref)*hbar] = 0

        E1 = E_roi - OM_T_roi*hbar
        E2 = E_roi - self.om_ref*hbar

        if self.ifWide:
            inferred_rho = rk.project_to_density_matrix(rho_raw,self.psd_sigma)
        else:

            gc.collect()

            amps_hat, mus_hat, sigmas_hat, betas_hat, taus_hat, lambdas_hat, gamma_hat, eta_hat = Bayesian_MCMC(
                E1.flatten(),
                E2.flatten(),
                rhodata_roi.flatten(),
                rhosigma_roi.flatten(),
                n_peaks=self.mcmc_peaks,
                num_warmup=self.mcmc_num_warmup, 
                num_samples=self.mcmc_num_samples, 
                num_chains=self.mcmc_num_chains,
                suffix=suffix
            )

            def rho_inferred(e1,e2):
                return rk.rho_model(e1,e2,amps_hat,mus_hat,sigmas_hat,betas_hat,taus_hat,lambdas_hat,gamma_hat,eta_hat)

            inferred_rho = rho_inferred(self.E1interp, self.E2interp)
            inferred_rho = inferred_rho/np.trace(inferred_rho)

        return rho_raw, rho_raw_sigma, inferred_rho
    
    def mcmc_fit(self):

        rho_raw, rho_raw_sigma, inferred_rho = self._apply_mcmc(self.rhodata, self.rhodata_roi, self.rhosigma, self.rhosigma_roi, self.E_roi, self.OM_T_roi, suffix='')

        ideal_rho = self.rho_f(self.E1interp + self.om_ref*hbar, self.E2interp + self.om_ref*hbar)
        ideal_rho = ideal_rho/np.trace(ideal_rho)

        # rk.plot_mat(ideal_rho - 1e-4, extent=[self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi], cmap='plasma',
        #         saveloc='single_output_temp/rhos/rho_ideal.png', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
        #         title='Initial photoelectron density matrix', show=False, square=True)
        np.savez('single_output_temp/6mcmc/rho_ideal.npz',
                 mat_complex=ideal_rho,extent=np.array([self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi]))
        
        # rk.plot_mat(rho_raw - 1e-4, extent=[self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi], cmap='plasma',
        #         saveloc='single_output_temp/rhos/rho_raw', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
        #         title='$\\tilde S_{corr}(\\varepsilon_2,\\varepsilon_1)$', show=False, square=True)
        np.savez('single_output_temp/6mcmc/data_rho_interp.npz',
                 mat_complex=rho_raw,extent=np.array([self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi]))
        
        # rk.plot_mat(rho_raw_sigma, extent=[self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi], cmap='plasma',
        #         saveloc='single_output_temp/rhos/rho_raw_sigma', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
        #         title='$\\sigma (\\varepsilon_2,\\varepsilon_1)$', show=False, mode='abs')
        np.savez('single_output_temp/6mcmc/data_sigma_interp.npz',
                 mat_abs=rho_raw_sigma,extent=np.array([self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi]))
        
        fid = rk.fidelity(ideal_rho, inferred_rho)

        # rk.plot_mat(inferred_rho - 1e-4, extent=[self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi], cmap='plasma',
        #         saveloc=f'single_output_temp/rhos/rho_inferred.png', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
        #         title=f'Inferred density matrix', show=False, caption='F=%.3f'%fid, square=True)
        np.savez('single_output_temp/6mcmc/rho_inferred.npz',
                 mat_complex=inferred_rho,extent=np.array([self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi]),RES=fid)
        
        rho_raw_rec, rho_raw_sigma_rec, inferred_rho_rec = self._apply_mcmc(self.rhodata_rec, self.rhodata_roi_rec, 
                                        self.rhosigma_rec, self.rhosigma_roi_rec, self.E_roi_rec, self.OM_T_roi_rec, suffix='_rec')

        # rk.plot_mat(rho_raw_rec - 1e-4, extent=[self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi], cmap='plasma',
        #         saveloc='single_output_temp/rhos/rho_raw_rec', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
        #         title='$\\tilde S_{corr}(\\varepsilon_2,\\varepsilon_1)$', show=False, square=True)
        np.savez('single_output_temp/6mcmc/data_rho_interp_rec.npz',
                 mat_complex=rho_raw_rec,extent=np.array([self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi]))

        # rk.plot_mat(rho_raw_sigma_rec, extent=[self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi], cmap='plasma',
        #         saveloc='single_output_temp/rhos/rho_raw_sigma_rec', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
        #         title='$\\sigma (\\varepsilon_2,\\varepsilon_1)$', show=False, mode='abs')
        np.savez('single_output_temp/6mcmc/data_sigma_interp_rec.npz',
                 mat_abs=rho_raw_sigma_rec,extent=np.array([self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi]))
        
        fid_rec = rk.fidelity(ideal_rho, inferred_rho_rec)

        # rk.plot_mat(inferred_rho_rec - 1e-4, extent=[self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi], cmap='plasma',
        #         saveloc=f'single_output_temp/rhos/rho_inferred_rec.png', xlabel='Energy $\\varepsilon_2$ [eV]', ylabel='Energy $\\varepsilon_1$ [eV]',
        #         title=f'Inferred density matrix', show=False, caption='F=%.3f'%fid_rec, square=True)
        np.savez('single_output_temp/6mcmc/rho_inferred_rec.npz',
                 mat_complex=inferred_rho_rec,extent=np.array([self.harmq_lo,self.harmq_hi,self.harmq_lo,self.harmq_hi]),RES=fid_rec)
        
        return

    def run_full_pipeline(self):
        self.generate_signal()
        self.process_and_detrend()
        self.kb_correct()
        self.probe_reconstruct()
        self.probe_sp_correct()
        self.mcmc_fit()