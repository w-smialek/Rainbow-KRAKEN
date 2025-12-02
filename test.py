import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

hbar = 6.582119569e-1

def CFT(T_range, signal, use_window=True, inverse=False, zero_pad=0):
    """
    Continuous-time Fourier transform convention per column (fixed energy E):
        Forward: S(ω, E) = ∫ s(t, E) · e^{-i ω t} dt

    Discrete implementation with FFT for samples t_n = t0 + n·dt:
        S(ω_k, E) ≈ dt · e^{-i ω_k t0} · FFT_n{ s(t_n, E) }[k]

    Options:
      - use_window: apply Hann window (coherent-gain normalized) along time (rows).
      - inverse: when True, perform the inverse operation to reconstruct s(t, E)
                 from S(ω, E). The input "signal" must be the shifted spectrum
                 (as returned by the forward path). Windowing is not undone.
      - zero_pad: integer number of zeros to pad on BOTH sides along time before
                  transform (forward) or to account for when inverting. Default 0.

    Returns (forward, inverse respectively):
      - forward: spec_shift, OM_T, E_min, E_max
      - inverse: sig_time (trimmed to len(T_range) along rows), None, t_min, t_max
    """
    # Validate input and infer sizes locally (avoid globals)
    t = np.asarray(T_range)
    if t.ndim != 1:
        raise ValueError("T_range must be a 1D array of time samples")
    NT_local = t.size
    NE_local = signal.shape[1]

    # Ensure uniform sampling
    if NT_local < 2:
        raise ValueError("T_range must contain at least two samples")
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=0.0):
        raise ValueError("T_range must be uniformly spaced")

    # Inverse path: reconstruct time-domain from shifted spectrum
    if inverse:
        # Infer padded length from provided spectrum along rows
        S_shift = np.asarray(signal)
        if S_shift.ndim != 2:
            raise ValueError("For inverse, 'signal' must be a 2D array with rows as ω bins")
        N_eff = S_shift.shape[0]

        # Frequency grid associated with the padded length
        freqs = np.fft.fftfreq(N_eff, d=dt)
        omega = 2 * np.pi * freqs

        # Undo shift and the forward phase factor (use padded start time)
        t0_eff = t[0] - int(zero_pad) * dt
        S_unshift = np.fft.ifftshift(S_shift, axes=0)
        phase_inv = np.exp(1j * omega[:, None] * t0_eff)
        S_unphased = S_unshift * phase_inv

        # Inverse scaling: x ≈ (1/dt) · IFFT{ S }
        sig_time_full = np.fft.ifft(S_unphased, axis=0) / dt

        # Trim padding back to original time length
        zp = int(zero_pad)
        if zp > 0:
            lo = zp
            hi = zp + NT_local
            sig_time = sig_time_full[lo:hi, :]
        else:
            sig_time = sig_time_full

        # Return time-domain signal and simple time metadata
        return sig_time, None, float(t[0]), float(t[-1])

    # Forward path
    if signal.shape[0] != NT_local:
        raise ValueError("signal shape[0] must match len(T_range)")

    # Windowing (apply on original length), then symmetric zero padding
    if use_window:
        window = np.hanning(NT_local)[:, None]
        coherent_gain = window.mean()  # Hann coherent gain = 0.5
    else:
        window = np.ones((NT_local, 1))
        coherent_gain = 1.0

    windowed = (signal * window) / coherent_gain

    zp = int(zero_pad)
    if zp > 0:
        windowed = np.pad(windowed, ((zp, zp), (0, 0)), mode='constant', constant_values=0)
    N_eff = windowed.shape[0]

    # Frequencies (unshifted) and corresponding angular frequencies for padded length
    freqs = np.fft.fftfreq(N_eff, d=dt)   # cycles per unit time
    omega = 2 * np.pi * freqs             # rad / unit time

    # FFT along time axis with dt scaling to approximate the continuous integral
    spec = np.fft.fft(windowed, axis=0) * dt

    # Phase correction for start time including leading padding
    t0_eff = t[0] - zp * dt
    phase = np.exp(-1j * omega[:, None] * t0_eff)
    spec *= phase

    # Shift spectrum and build energy axis (ħ·ω)
    spec_shift = np.fft.fftshift(spec, axes=0)
    energy_axis = hbar * omega
    energy_axis_shift = np.fft.fftshift(energy_axis)

    OM_T = (energy_axis_shift / hbar)  # back to angular frequency ω
    OM_T = np.tile(OM_T, (NE_local, 1)).T

    return spec_shift, OM_T, energy_axis_shift[0], energy_axis_shift[-1]


def plot_mat(mat,extent=[0,1,0,1],cmap='plasma',mode='phase',show=True,saveloc=None,caption=None,xlabel='x',ylabel='y',title=None):

    if mode == 'abs':
        plt.figure()
        im = plt.imshow(np.abs(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        else:
            plt.title('|M|')
        plt.colorbar(im)
        if caption is not None:
            plt.text(0.02, 0.98, caption, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='left',
                    color='white', weight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    elif mode == 'phase':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        im0 = axes[0].imshow(np.abs(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[0].set_title('|M|')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        fig.colorbar(im0, ax=axes[0])
        if caption is not None:
            axes[0].text(0.02, 0.98, caption, transform=axes[0].transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='left',
                        color='white', weight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        im1 = axes[1].imshow(np.angle(mat), extent=extent, origin='lower', aspect='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('arg(M)')
        axes[1].set_xlabel(xlabel)
        fig.colorbar(im1, ax=axes[1])

        if title:
            fig.suptitle(title)

        plt.tight_layout()

    elif mode == 'reim':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        im0 = axes[0].imshow(np.real(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[0].set_title('Re{M}')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        fig.colorbar(im0, ax=axes[0])
        if caption is not None:
            axes[0].text(0.02, 0.98, caption, transform=axes[0].transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='left',
                        color='white', weight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        im1 = axes[1].imshow(np.imag(mat), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        axes[1].set_title('Im{M}')
        axes[1].set_xlabel(xlabel)
        fig.colorbar(im1, ax=axes[1])

        if title:
            fig.suptitle(title)

        plt.tight_layout()

    if saveloc:
        plt.savefig(saveloc,dpi=400)
    if show:
        plt.show()
    plt.close()
    return

mus = []
valar = np.linspace(63.5,65,10)
for E_hi in valar:

    E_lo = 60.0
    E_hi = 63.5
    T_reach = 100
    E_span = E_hi-E_lo

    N_E = 1000
    N_T = 1000
    N_T = int(N_T)
    N_E = int(N_E)

    E_range = np.linspace(E_lo,E_hi,N_E)
    T_range = np.linspace(-T_reach,T_reach,N_T)
    E, T = np.meshgrid(E_range,T_range)

    alpha = 500
    sigma = 0.5
    sig = alpha*np.exp(-((E-E[0,N_E//2])/sigma)**2)

    rng = np.random.default_rng()
    signal = rng.poisson(sig).astype(float)

    signal_FT, OM_T, em_lo, em_hi = CFT(T_range,signal,use_window=False)

    # plt.plot(np.mean(np.abs(signal_FT[:400,:])**2,axis=0))
    # plt.plot(np.sum(signal,axis=0))
    # plt.show()

    ratio = np.sum(signal,axis=0)*(2*T_reach/N_T)**2 /  (np.mean(np.abs(signal_FT[:400,:])**2,axis=0)+1e-10)
    mu = np.mean(ratio)
    pm = np.sqrt(np.var(ratio))
    print(mu)

    mus.append(mu)

plt.plot(valar,mus)
plt.show()