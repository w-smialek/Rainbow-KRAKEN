import numpy as np
import matplotlib.pyplot as plt
from math import floor
from scipy.signal import fftconvolve
from matplotlib import patheffects as pe
from scipy.ndimage import gaussian_filter
from scipy.special import i0e, i1e
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import sqrtm

# Reduced Planck constant in eV*fs (approx CODATA): hbar = 6.582119569e-16 eV*s
hbar = 6.582119569e-1
from numpy import pi


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


def CFT(T_range, signal, use_window=True, inverse=False, zero_pad=0):
    """Continuous-time Fourier transform convention per column.

    Forward:
        S(omega, E) = integral s(t, E) * exp(-i * omega * t) dt

    Inverse:
        reconstruct time-domain signal from shifted spectrum.
    """
    t = np.asarray(T_range)
    if t.ndim != 1:
        raise ValueError("T_range must be a 1D array of time samples")
    NT_local = t.size
    NE_local = signal.shape[1]

    if NT_local < 2:
        raise ValueError("T_range must contain at least two samples")
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=0.0):
        raise ValueError("T_range must be uniformly spaced")

    if inverse:
        S_shift = np.asarray(signal)
        if S_shift.ndim != 2:
            raise ValueError("For inverse, 'signal' must be a 2D array with rows as omega bins")
        N_eff = S_shift.shape[0]

        freqs = np.fft.fftfreq(N_eff, d=dt)
        omega = 2 * np.pi * freqs

        t0_eff = t[0] - int(zero_pad) * dt
        S_unshift = np.fft.ifftshift(S_shift, axes=0)
        phase_inv = np.exp(1j * omega[:, None] * t0_eff)
        S_unphased = S_unshift * phase_inv

        sig_time_full = np.fft.ifft(S_unphased, axis=0) / dt

        zp = int(zero_pad)
        if zp > 0:
            lo = zp
            hi = zp + NT_local
            sig_time = sig_time_full[lo:hi, :]
        else:
            sig_time = sig_time_full

        return sig_time, None, float(t[0]), float(t[-1])

    if signal.shape[0] != NT_local:
        raise ValueError("signal shape[0] must match len(T_range)")

    if use_window:
        window = np.hanning(NT_local)[:, None]
    else:
        window = np.ones((NT_local, 1))

    windowed = signal * window

    zp = int(zero_pad)
    if zp > 0:
        windowed = np.pad(windowed, ((zp, zp), (0, 0)), mode="constant", constant_values=0)
    N_eff = windowed.shape[0]

    freqs = np.fft.fftfreq(N_eff, d=dt)
    omega = 2 * np.pi * freqs

    spec = np.fft.fft(windowed, axis=0) * dt

    t0_eff = t[0] - zp * dt
    phase = np.exp(-1j * omega[:, None] * t0_eff)
    spec *= phase

    spec_shift = np.fft.fftshift(spec, axes=0)
    energy_axis = hbar * omega
    energy_axis_shift = np.fft.fftshift(energy_axis)

    if zp > 0:
        trim_lo = (N_eff - NT_local) // 2
        trim_hi = trim_lo + NT_local
        spec_shift = spec_shift[trim_lo:trim_hi, :]
        energy_axis_shift = energy_axis_shift[trim_lo:trim_hi]

    OM_T = energy_axis_shift / hbar
    OM_T = np.tile(OM_T, (NE_local, 1)).T

    return spec_shift, OM_T, energy_axis_shift[0], energy_axis_shift[-1]


def _style_ax(ax, grid_alpha=None):
    """Apply compact axis styling used by plot_mat."""
    _tick_kw = dict(fontsize=9)
    ax.tick_params(
        axis="both",
        which="major",
        length=0,
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelsize=_tick_kw["fontsize"],
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if grid_alpha is not None:
        ax.grid(alpha=grid_alpha)


def _set_title_with_math_style(setter, text, title_kw, title_math_font="stix", title_math_bold=True):
    """Render title text with optional math-font and bold mathtext settings."""
    if text is None:
        return

    if "$" not in text:
        setter(text, **title_kw)
        return

    valid_fontsets = {"dejavusans", "dejavuserif", "cm", "stix", "stixsans", "custom"}
    if title_math_font is None:
        math_fontset = plt.rcParams.get("mathtext.fontset", "dejavusans")
    else:
        math_fontset = str(title_math_font).lower()
        if math_fontset not in valid_fontsets:
            raise ValueError(
                f"title_math_font must be one of {sorted(valid_fontsets)} or None, got {title_math_font!r}"
            )

    rc_updates = {"mathtext.fontset": math_fontset}
    if title_math_bold:
        rc_updates["mathtext.default"] = "bf"

    with plt.rc_context(rc_updates):
        setter(text, **title_kw)


def plot_mat(
    mat,
    extent=[0, 1, 0, 1],
    cmap="plasma",
    mode="phase",
    show=True,
    saveloc=None,
    caption=None,
    xlabel="x",
    ylabel="y",
    title=None,
    grid_alpha=None,
    title_math_bold=True,
    title_math_font="dejavusans",
    square=False,
):
    _label_kw = dict(fontsize=10)
    _title_kw0 = dict(fontsize=10, fontweight="semibold")
    _title_kwabs = dict(fontsize=15, fontweight="semibold", y=1.03)
    _title_kw = dict(fontsize=15, fontweight="semibold")

    if mode == "abs":
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im = ax.imshow(np.abs(mat), extent=extent, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xlabel(xlabel, **_label_kw)
        ax.set_ylabel(ylabel, **_label_kw)
        if title:
            _set_title_with_math_style(
                ax.set_title,
                title,
                _title_kwabs,
                title_math_font=title_math_font,
                title_math_bold=title_math_bold,
            )
        else:
            ax.set_title("Abs(M)", **_title_kwabs)
        fig.colorbar(im, ax=ax)
        if caption is not None:
            ax.text(
                0.02,
                0.98,
                caption,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                color="white",
                weight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

        _style_ax(ax, grid_alpha=grid_alpha)
        plt.tight_layout()

    elif mode == "phase":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        if square:
            axes[0].set_box_aspect(1)
            axes[1].set_box_aspect(1)
            axes[0].set_anchor("E")
            axes[1].set_anchor("W")

        im0 = axes[0].imshow(np.abs(mat), extent=extent, origin="lower", aspect="auto", cmap=cmap)
        axes[0].set_title("Abs(M)", **_title_kw0)
        axes[0].set_xlabel(xlabel, **_label_kw)
        axes[0].set_ylabel(ylabel, **_label_kw)
        fig.colorbar(im0, ax=axes[0])
        if caption is not None:
            axes[0].text(
                0.02,
                0.98,
                caption,
                transform=axes[0].transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                color="white",
                weight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

        im1 = axes[1].imshow(
            np.angle(mat),
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[1].set_title("arg(M)", **_title_kw0)
        axes[1].set_xlabel(xlabel, **_label_kw)
        fig.colorbar(im1, ax=axes[1])

        if title:
            title_kw = dict(_title_kw)
            if square:
                title_kw["y"] = 1.01
            _set_title_with_math_style(
                fig.suptitle,
                title,
                title_kw,
                title_math_font=title_math_font,
                title_math_bold=title_math_bold,
            )

        for ax in axes:
            _style_ax(ax, grid_alpha=grid_alpha)

        if square:
            fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.87, wspace=0.08)
        else:
            plt.tight_layout()

    if saveloc:
        plt.savefig(saveloc, dpi=500, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def spectrum_fun(A, om0, s, om):
    retval = A / (2 * s) * np.exp(-(om - om0) ** 2 / (2 * s**2))
    return retval


def extract_midslice(sig_full, slice_fracts, sliced_range, e_slice_fracts=None, e_sliced_range=None):
    if e_slice_fracts is None:
        n_t, n_e = sig_full.shape
        mid_emrange_lo, mid_emrange_hi = slice_fracts

        i0 = floor(n_t * mid_emrange_lo)
        i1 = floor(n_t * mid_emrange_hi) + 1

        axis_mid = sliced_range[i0:i1]
        sig_mid = sig_full[i0:i1, :]

        return sig_mid, axis_mid, i0, i1

    n_t, n_e = sig_full.shape
    mid_emrange_lo, mid_emrange_hi = slice_fracts
    mid_erange_lo, mid_erange_hi = e_slice_fracts

    i0 = floor(n_t * mid_emrange_lo)
    i1 = floor(n_t * mid_emrange_hi) + 1

    e_i0 = floor(n_e * mid_erange_lo)
    e_i1 = floor(n_e * mid_erange_hi) + 1

    axis_mid = sliced_range[i0:i1]
    e_axis_mid = e_sliced_range[e_i0:e_i1]
    sig_mid = sig_full[i0:i1, e_i0:e_i1]

    return sig_mid, axis_mid, i0, i1, e_axis_mid, e_i0, e_i1


def koay_basser_correction(M_obs, mean_noise_power, lambda_thresh=1, only_floor=False):
    """Remove Rician bias using Koay-Basser inversion."""
    sigma = np.sqrt(mean_noise_power / 2.0)
    noise_floor = sigma * np.sqrt(np.pi / 2.0)

    M_obs = np.atleast_1d(M_obs)
    A_est = np.zeros_like(M_obs)

    def rician_mean_eq(snr_a, snr_m_target):
        y = (snr_a**2) / 4.0
        f_val = np.sqrt(np.pi / 2.0) * ((1 + 2 * y) * i0e(y) + 2 * y * i1e(y))
        return f_val - snr_m_target

    for i, m in enumerate(M_obs):
        if m <= lambda_thresh * noise_floor:
            A_est[i] = 0.0
        else:
            if only_floor:
                A_est[i] = m
            else:
                current_snr_m = m / sigma
                root_snr = brentq(rician_mean_eq, 0, current_snr_m * 2, args=(current_snr_m,))
                A_est[i] = root_snr * sigma

    return A_est if len(A_est) > 1 else A_est[0]


def new_Sig_cc_interp(re_spline, im_spline, om_t_vals, Ef_vals, grid=None):
    """Evaluate cubic splines on either grid axes or pointwise arrays."""
    om_t_vals = np.asarray(om_t_vals)
    Ef_vals = np.asarray(Ef_vals)

    if om_t_vals.ndim == 1 and Ef_vals.ndim == 1 and (grid is None or grid is True):
        re = re_spline(om_t_vals, Ef_vals, grid=True)
        im = im_spline(om_t_vals, Ef_vals, grid=True)
        return re + 1j * im

    om_b, ef_b = np.broadcast_arrays(om_t_vals, Ef_vals)
    re = re_spline.ev(om_b.ravel(), ef_b.ravel()).reshape(om_b.shape)
    im = im_spline.ev(om_b.ravel(), ef_b.ravel()).reshape(om_b.shape)
    return re + 1j * im


def resample(spec_corrected, rho_hi, rho_lo, om_ref, E, OM_T, N_NEW):
    rho_valrange = rho_hi - rho_lo

    new_Ef_min = rho_lo + hbar * om_ref
    new_Ef_max = rho_hi + hbar * om_ref

    new_omt_min = om_ref - rho_valrange / hbar
    new_omt_max = om_ref + rho_valrange / hbar

    idx_min = np.argmin(np.abs(new_Ef_min - E[0, :]))
    idx_max = np.argmin(np.abs(new_Ef_max - E[0, :]))

    idy_min = np.argmin(np.abs(new_omt_min - OM_T[:, 0]))
    idy_max = np.argmin(np.abs(new_omt_max - OM_T[:, 0]))

    extent = [E[0, idx_min], E[0, idx_max], hbar * OM_T[idy_min, 0], hbar * OM_T[idy_max, 0]]

    small_sig = spec_corrected[idy_min:idy_max, idx_min:idx_max]

    Ef_range = E[0, idx_min:idx_max]
    omt_range = OM_T[idy_min:idy_max, 0]

    re_spline = RectBivariateSpline(omt_range * hbar, Ef_range, np.real(small_sig), kx=3, ky=3, s=0)
    im_spline = RectBivariateSpline(omt_range * hbar, Ef_range, np.imag(small_sig), kx=3, ky=3, s=0)

    e1_range = np.linspace(rho_lo, rho_hi, N_NEW)
    e2_range = np.linspace(rho_lo, rho_hi, N_NEW)

    E1, E2 = np.meshgrid(e1_range, e2_range, indexing="ij")

    EPS1 = E2 - E1 + hbar * om_ref
    EPS2 = E2 + hbar * om_ref

    Sig_cc_cubic_mesh = new_Sig_cc_interp(re_spline, im_spline, EPS1, EPS2)

    return Sig_cc_cubic_mesh, small_sig, extent, [idy_min, idy_max, idx_min, idx_max], E1, E2


def project_to_density_matrix(M, smooth_sigma=1.5):
    """Project a noisy measured matrix onto Hermitian PSD trace-1 matrices."""
    d = M.shape[0]

    H = (M + M.conj().T) / 2

    if smooth_sigma > 0:
        H = gaussian_filter(np.real(H), sigma=smooth_sigma) + 1j * gaussian_filter(
            np.imag(H), sigma=smooth_sigma
        )
        H = (H + H.conj().T) / 2

    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals, 0.0)

    eig_sum = np.sum(eigvals)
    if eig_sum <= 0:
        return np.eye(d, dtype=complex) / d
    eigvals = eigvals / eig_sum

    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho


def fidelity(rho, sigma):
    """Compute the Uhlmann fidelity between two density matrices."""
    rho = (rho + rho.conj().T) / 2
    sigma = (sigma + sigma.conj().T) / 2

    sqrt_rho = sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    fid = np.real((np.trace(sqrtm(inner))) ** 2)

    return float(fid)


def regularize_omega(om):
    """Regularize omega values near zero crossing to avoid instabilities."""
    om = np.asarray(om)
    result = np.copy(om)

    if om.ndim == 1:
        d_om = np.abs(om[1] - om[0]) if len(om) > 1 else 1.0
    elif om.ndim == 2:
        d_om = np.abs(om[1, 0] - om[0, 0]) if om.shape[0] > 1 else 1.0
    else:
        d_om = 1.0

    threshold = 10 * d_om
    close_to_zero = np.abs(om) <= threshold

    if np.any(close_to_zero):
        before_zero = (om < 0) & close_to_zero
        after_zero = (om >= 0) & close_to_zero

        result[before_zero] = -threshold
        result[after_zero] = threshold

    return result


def conv_bounds(result_lo, result_hi, N, s1_center):
    """Compute signal axis bounds for fftconvolve(..., mode='same')."""
    L = result_hi - result_lo
    dx = L / (N - 1)
    sum_lo = result_lo - ((N - 1) // 2) * dx

    s1_lo = s1_center - L / 2
    s2_lo = sum_lo - s1_lo
    s1_hi = s1_lo + L
    s2_hi = s2_lo + L
    return s1_lo, s2_lo, s1_hi, s2_hi