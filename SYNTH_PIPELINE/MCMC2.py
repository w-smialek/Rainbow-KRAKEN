import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import matplotlib.pyplot as plt
import os

hbar = 6.582119569e-1


def _circular_mean(angles, axis=0):
    """Return circular mean for angles in radians, robust to +/-pi wrap."""
    angles = np.asarray(angles)
    if angles.size == 0:
        return np.array([])
    return np.angle(np.mean(np.exp(1j * angles), axis=axis))

def rho_model(e1, e2, amps, mus, sigmas, a_s, bs, gammas):
    # Vectorized over peak indices to avoid Python O(n_peaks^2) loops inside NUTS.
    e1 = jnp.asarray(e1)
    e2 = jnp.asarray(e2)
    amps = jnp.asarray(amps)
    mus = jnp.asarray(mus)
    sigmas = jnp.asarray(sigmas)
    a_s = jnp.asarray(a_s)
    bs = jnp.asarray(bs)
    gammas = jnp.asarray(gammas)

    ndim = e1.ndim
    pshape = (slice(None),) + (None,) * ndim

    e1k = jnp.expand_dims(e1, axis=0)
    e2k = jnp.expand_dims(e2, axis=0)

    amps_k = amps[pshape]
    mus_k = mus[pshape]
    sigmas_k = sigmas[pshape]
    a_k = a_s[pshape]
    b_k = bs[pshape]

    x1 = e1k - mus_k
    x2 = e2k - mus_k

    rho1 = amps_k / (2.0 * sigmas_k) * jnp.exp(
        -(x1 ** 2) / (2.0 * sigmas_k ** 2) + 1j * (a_k * x1 ** 2 + b_k * x1)
    )
    rho2 = amps_k / (2.0 * sigmas_k) * jnp.exp(
        -(x2 ** 2) / (2.0 * sigmas_k ** 2) + 1j * (a_k * x2 ** 2 + b_k * x2)
    )

    # Keep gamma indices aligned with (k, l) contraction; no anti-diagonal swap.
    gamma_e = gammas[(slice(None), slice(None)) + (None,) * ndim]

    pair_terms = gamma_e * rho1[:, None, ...] * jnp.conj(rho2)[None, :, ...]
    return jnp.sum(pair_terms, axis=(0, 1))

def _build_hermitian_gamma(n_peaks, gamma_mag, gamma_phase):
    gamma = jnp.eye(n_peaks, dtype=jnp.complex128)
    if n_peaks <= 1:
        return gamma

    upper_i, upper_j = jnp.triu_indices(n_peaks, k=1)
    gij = gamma_mag * jnp.exp(1j * gamma_phase)
    gamma = gamma.at[upper_i, upper_j].set(gij)
    gamma = gamma.at[upper_j, upper_i].set(jnp.conj(gij))
    return gamma

# om_ref = 1.55 / hbar
# amps = [1.0 / np.sqrt(2), 1.0]
# mus = [25.0 - 0.17 + om_ref * hbar, 25.0 + om_ref * hbar]
# sigmas = [0.1, 0.1]
# a_s = [0, 4]
# bs = [3, 0]
# cs = [0, 0]
# Lambdas = [10, 10]
# gammas = np.array([[1.0, 0.0],
#                    [0.0, 1.0]])

def model(x, y, sigma_obs, z_obs=None, n_peaks=2):
    """
    Numpyro model for the 2D Gaussian density matrix with complex phase.
    """
    # Priors per peak (same prior family for each component).
    amps_now = numpyro.sample(
        'amps', dist.Uniform(0.0, 1000.0).expand([n_peaks]).to_event(1))
    mus_now = numpyro.sample(
        'mus', dist.Uniform(24.0, 26.0).expand([n_peaks]).to_event(1))
    sigmas_now = numpyro.sample(
        'sigmas', dist.HalfNormal(0.3).expand([n_peaks]).to_event(1))
    a_now = numpyro.sample(
        'a_now', dist.Normal(0.0, 8.0).expand([n_peaks]).to_event(1))
    b_now = numpyro.sample(
        'b_now', dist.Normal(0.0, 8.0).expand([n_peaks]).to_event(1))

    # Complex Hermitian gamma with unit diagonal and free off-diagonal entries.
    n_pairs = n_peaks * (n_peaks - 1) // 2
    if n_pairs > 0:
        gamma_mag = numpyro.sample(
            'gamma_mag', dist.Uniform(0.0, 1.0).expand([n_pairs]).to_event(1))
        gamma_phase = numpyro.sample(
            'gamma_phase', dist.Uniform(-jnp.pi, jnp.pi).expand([n_pairs]).to_event(1))
        gammas_now = _build_hermitian_gamma(n_peaks, gamma_mag, gamma_phase)
    else:
        gammas_now = jnp.eye(1, dtype=jnp.complex128)

    z_expected = rho_model(
        x,
        y,
        amps_now,
        mus_now,
        sigmas_now,
        a_now,
        b_now,
        gammas_now,
    )

    z_real_expected = jnp.real(z_expected)
    z_imag_expected = jnp.imag(z_expected)
    sigma_eff = jnp.clip(sigma_obs, 1e-12)
    
    # Use provided per-observation uncertainties for both real and imaginary parts.
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs_real', dist.Normal(z_real_expected, sigma_eff), obs=jnp.real(z_obs) if z_obs is not None else None)
        numpyro.sample('obs_imag', dist.Normal(z_imag_expected, sigma_eff), obs=jnp.imag(z_obs) if z_obs is not None else None)

def Bayesian_MCMC(x_obs, y_obs, z_obs, sigma_obs, n_peaks=2):
    print("Generating synthetic complex data...")
    # Generate data with known parameters
    
    print("\nSetting up NUTS MCMC...")
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=1)
    
    print("Running MCMC to obtain posterior distributions...")
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(
        rng_key, 
        x=jnp.array(x_obs), 
        y=jnp.array(y_obs), 
        sigma_obs=jnp.array(sigma_obs),
        z_obs=jnp.array(z_obs),
        n_peaks=n_peaks,
    )
    
    print("\nMCMC Summary:")
    mcmc.print_summary()
    
    # Extract posterior samples
    samples = mcmc.get_samples()
    amps_samples = np.array(samples['amps'])
    mus_samples = np.array(samples['mus'])
    sigmas_samples = np.array(samples['sigmas'])
    a_samples = np.array(samples['a_now'])
    b_samples = np.array(samples['b_now'])
    gamma_mag_samples = np.array(samples['gamma_mag']) if n_peaks > 1 else np.zeros((amps_samples.shape[0], 0))
    gamma_phase_samples = np.array(samples['gamma_phase']) if n_peaks > 1 else np.zeros((amps_samples.shape[0], 0))

    amps_hat = np.mean(amps_samples, axis=0)
    mus_hat = np.mean(mus_samples, axis=0)
    sigmas_hat = np.maximum(np.mean(sigmas_samples, axis=0), 1e-8)
    a_hat = np.mean(a_samples, axis=0)
    b_hat = np.mean(b_samples, axis=0)

    if n_peaks > 1:
        gamma_complex_samples = gamma_mag_samples * np.exp(1j * gamma_phase_samples)
        gamma_complex_hat = np.mean(gamma_complex_samples, axis=0)
        gamma_mag_hat = np.abs(gamma_complex_hat)
        gamma_phase_hat = np.angle(gamma_complex_hat)
    else:
        gamma_mag_hat = np.array([])
        gamma_phase_hat = np.array([])
    
    # Plot posterior diagnostics for all fitted parameters.
    posterior_panels = []

    for k in range(n_peaks):
        posterior_panels.append((f'amps[{k}]', amps_samples[:, k]))
        posterior_panels.append((f'mus[{k}]', mus_samples[:, k]))
        posterior_panels.append((f'sigmas[{k}]', sigmas_samples[:, k]))
        posterior_panels.append((f'a_s[{k}]', a_samples[:, k]))
        posterior_panels.append((f'b_s[{k}]', b_samples[:, k]))

    pair_idx = 0
    for i in range(n_peaks):
        for j in range(i + 1, n_peaks):
            posterior_panels.append((f'|gamma[{i},{j}]|', gamma_mag_samples[:, pair_idx]))
            posterior_panels.append((f'arg(gamma[{i},{j}])', gamma_phase_samples[:, pair_idx]))
            pair_idx += 1

    n_panels = len(posterior_panels)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.8 * n_rows))
    axs = np.atleast_1d(axs).ravel()

    for ax, (label, values) in zip(axs, posterior_panels):
        if label.startswith('arg('):
            values_wrapped = ((values + np.pi) % (2.0 * np.pi)) - np.pi
            ax.hist(values_wrapped, bins=np.linspace(-np.pi, np.pi, 41), density=True, alpha=0.75, color='tab:blue')
            ax.axvline(_circular_mean(values_wrapped), color='black', linestyle='--', linewidth=1.0)
            ax.set_xlim(-np.pi, np.pi)
        else:
            ax.hist(values, bins=40, density=True, alpha=0.75, color='tab:blue')
            ax.axvline(np.mean(values), color='black', linestyle='--', linewidth=1.0)
        ax.set_title(label, fontsize=10)

    for ax in axs[n_panels:]:
        ax.axis('off')

    fig.suptitle('Posterior distributions of fitted parameters', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('MCMC_out/out1.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

    # Observed data diagnostics mapped to a full grid; unobserved entries stay at zero.
    z_amp_obs = np.abs(z_obs)
    z_phase_obs = np.angle(z_obs)
    x_grid = np.unique(np.sort(x_obs))
    y_grid = np.unique(np.sort(y_obs))
    nx_obs, ny_obs = len(x_grid), len(y_grid)

    amp_grid = np.zeros((ny_obs, nx_obs), dtype=float)
    phase_grid = np.zeros((ny_obs, nx_obs), dtype=float)
    sigma_grid = np.zeros((ny_obs, nx_obs), dtype=float)

    xi = np.searchsorted(x_grid, x_obs)
    yi = np.searchsorted(y_grid, y_obs)
    amp_grid[yi, xi] = z_amp_obs
    phase_grid[yi, xi] = z_phase_obs
    sigma_grid[yi, xi] = sigma_obs

    fig_obs, axs_obs = plt.subplots(1, 3, figsize=(18, 5))
    extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]

    im_amp_obs = axs_obs[0].imshow(amp_grid, origin='lower', extent=extent, aspect='auto', cmap='plasma')
    axs_obs[0].set_title('Observed |z_obs| (missing=0)')
    axs_obs[0].set_xlabel('x_obs')
    axs_obs[0].set_ylabel('y_obs')
    fig_obs.colorbar(im_amp_obs, ax=axs_obs[0], label='Amplitude')

    im_phase_obs = axs_obs[1].imshow(phase_grid, origin='lower', extent=extent, aspect='auto', cmap='hsv')
    axs_obs[1].set_title('Observed phase(z_obs) (missing=0)')
    axs_obs[1].set_xlabel('x_obs')
    axs_obs[1].set_ylabel('y_obs')
    fig_obs.colorbar(im_phase_obs, ax=axs_obs[1], label='Phase [rad]')

    im_sigma_obs = axs_obs[2].imshow(sigma_grid, origin='lower', extent=extent, aspect='auto', cmap='magma')
    axs_obs[2].set_title('Provided sigma_obs (missing=0)')
    axs_obs[2].set_xlabel('x_obs')
    axs_obs[2].set_ylabel('y_obs')
    fig_obs.colorbar(im_sigma_obs, ax=axs_obs[2], label='Sigma')

    plt.tight_layout()
    plt.savefig('MCMC_out/diagnostic_observed.png')
    plt.close(fig_obs)

    # Final inferred matrix from posterior-mean parameters.
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    gamma_hat = _build_hermitian_gamma(n_peaks, jnp.array(gamma_mag_hat), jnp.array(gamma_phase_hat))


    print(  jnp.array(amps_hat),
            jnp.array(mus_hat),
            jnp.array(sigmas_hat),
            jnp.array(a_hat),
            jnp.array(b_hat),
            gamma_hat)

    z_inf = rho_model(
        jnp.array(Xg),
        jnp.array(Yg),
        jnp.array(amps_hat),
        jnp.array(mus_hat),
        jnp.array(sigmas_hat),
        jnp.array(a_hat),
        jnp.array(b_hat),
        gamma_hat,
    )
    amp_inf = np.abs(np.array(z_inf))
    phase_inf = np.angle(np.array(z_inf))

    fig_inf, axs_inf = plt.subplots(1, 2, figsize=(12, 5))
    im_amp = axs_inf[0].imshow(amp_inf, origin='lower', extent=extent, aspect='auto', cmap='plasma')
    axs_inf[0].set_title('Inferred matrix amplitude')
    axs_inf[0].set_xlabel('x')
    axs_inf[0].set_ylabel('y')
    fig_inf.colorbar(im_amp, ax=axs_inf[0], label='Amplitude')

    im_phase = axs_inf[1].imshow(phase_inf, origin='lower', extent=extent, aspect='auto', cmap='hsv')
    axs_inf[1].set_title('Inferred matrix phase')
    axs_inf[1].set_xlabel('x')
    axs_inf[1].set_ylabel('y')
    fig_inf.colorbar(im_phase, ax=axs_inf[1], label='Phase [rad]')

    plt.tight_layout()
    plt.savefig('MCMC_out/inferred_matrix.png')
    plt.close(fig_inf)