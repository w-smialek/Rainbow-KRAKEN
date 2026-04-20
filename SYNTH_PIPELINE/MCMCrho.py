import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import matplotlib.pyplot as plt

def _circular_mean(angles, axis=0):
    """Return circular mean for angles in radians, robust to +/-pi wrap."""
    angles = np.asarray(angles)
    if angles.size == 0:
        return np.array([])
    return np.angle(np.mean(np.exp(1j * angles), axis=axis))

def rho_model(e1, e2, amps, mus, sigmas, betas, taus, lambdas, gammas, etas):
    # Vectorized over peak indices to avoid Python O(n_peaks^2) loops inside NUTS.
    e1 = jnp.asarray(e1)
    e2 = jnp.asarray(e2)
    amps = jnp.asarray(amps)
    mus = jnp.asarray(mus)
    sigmas = jnp.asarray(sigmas)
    betas = jnp.asarray(betas)
    taus = jnp.asarray(taus)
    lambdas = jnp.asarray(lambdas)
    gammas = jnp.asarray(gammas)
    etas = jnp.asarray(etas)

    ndim = e1.ndim
    pshape = (slice(None),) + (None,) * ndim

    e1k = jnp.expand_dims(e1, axis=0)
    e2k = jnp.expand_dims(e2, axis=0)

    amps_k = amps[pshape]
    mus_k = mus[pshape]
    sigmas_k = sigmas[pshape]
    beta_k = betas[pshape]
    tau_k = taus[pshape]
    lambda_k = lambdas[pshape]

    x1 = e1k - mus_k
    x2 = e2k - mus_k

    rho1 = amps_k / (2.0 * jnp.pi * sigmas_k ** 2)**(0.25) * jnp.exp(
        -(x1 ** 2) * (1/(4.0 * sigmas_k ** 2) - 1j*beta_k/2) + 1j * tau_k * x1
    )
    rho2 = amps_k / (2.0 * jnp.pi * sigmas_k ** 2)**(0.25) * jnp.exp(
        -(x2 ** 2) * (1/(4.0 * sigmas_k ** 2) - 1j*beta_k/2) + 1j * tau_k * x2
    )

    # Keep gamma indices aligned with (k, l) contraction; no anti-diagonal swap.
    gamma_e = gammas[(slice(None), slice(None)) + (None,) * ndim]
    eta_e = etas[(slice(None), slice(None)) + (None,) * ndim]

    D_kl = jnp.exp(
        - lambda_k[:, None, ...]**2 / 2 * x1[:, None, ...]**2
        - lambda_k[None, :, ...]**2 / 2 * x2[None, :, ...]**2
        + eta_e * lambda_k[:, None, ...] * lambda_k[None, :, ...] * x1[:, None, ...] * x2[None, :, ...]
    )

    pair_terms = gamma_e * rho1[:, None, ...] * jnp.conj(rho2)[None, :, ...] * D_kl
    return jnp.sum(pair_terms, axis=(0, 1))

def _build_symmetric_eta(n_peaks, eta_off_diag):
    eta = jnp.eye(n_peaks, dtype=jnp.float64)
    if n_peaks <= 1:
        return eta

    upper_i, upper_j = jnp.triu_indices(n_peaks, k=1)
    eta = eta.at[upper_i, upper_j].set(eta_off_diag)
    eta = eta.at[upper_j, upper_i].set(eta_off_diag)
    return eta

def _build_hermitian_gamma(n_peaks, gamma_mag, gamma_phase):
    gamma = jnp.eye(n_peaks, dtype=jnp.complex128)
    if n_peaks <= 1:
        return gamma

    upper_i, upper_j = jnp.triu_indices(n_peaks, k=1)
    gij = gamma_mag * jnp.exp(1j * gamma_phase)
    gamma = gamma.at[upper_i, upper_j].set(gij)
    gamma = gamma.at[upper_j, upper_i].set(jnp.conj(gij))
    return gamma

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
        'sigmas', dist.Uniform(0.0,0.3).expand([n_peaks]).to_event(1))
    betas_now = numpyro.sample(
        'betas_now', dist.Normal(0.0, 8.0).expand([n_peaks]).to_event(1))
    taus_now = numpyro.sample(
        'taus_now', dist.Normal(0.0, 8.0).expand([n_peaks]).to_event(1))
    lambdas_now = numpyro.sample(
        'lambdas_now', dist.HalfNormal(4.0).expand([n_peaks]).to_event(1))

    # Complex Hermitian gamma with unit diagonal and free off-diagonal entries.
    n_pairs = n_peaks * (n_peaks - 1) // 2
    if n_pairs > 0:
        # gamma_mag = numpyro.sample(
        #     'gamma_mag', dist.Uniform(0.0, 1.0).expand([n_pairs]).to_event(1))
        gamma_mag = numpyro.sample(
            'gamma_mag', dist.HalfNormal(0.05).expand([n_pairs]).to_event(1))
        gamma_phase = numpyro.sample(
            'gamma_phase', dist.Uniform(-1.1*jnp.pi, 1.1*jnp.pi).expand([n_pairs]).to_event(1))
        gammas_now = _build_hermitian_gamma(n_peaks, gamma_mag, gamma_phase)
        eta_off_diag = numpyro.sample(
            'eta_off_diag', dist.Uniform(-1.0, 1.0).expand([n_pairs]).to_event(1))
        etas_now = _build_symmetric_eta(n_peaks, eta_off_diag)
    else:
        gammas_now = jnp.eye(1, dtype=jnp.complex128)
        etas_now = jnp.eye(1, dtype=jnp.float64)

    z_expected = rho_model(
        x,
        y,
        amps_now,
        mus_now,
        sigmas_now,
        betas_now,
        taus_now,
        lambdas_now,
        gammas_now,
        etas_now,
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
    # mcmc = MCMC(nuts_kernel, num_warmup=10, num_samples=20, num_chains=1)
    
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
    betas_samples = np.array(samples['betas_now'])
    taus_samples = np.array(samples['taus_now'])
    lambdas_samples = np.array(samples['lambdas_now'])
    gamma_mag_samples = np.array(samples['gamma_mag']) if n_peaks > 1 else np.zeros((amps_samples.shape[0], 0))
    gamma_phase_samples = np.array(samples['gamma_phase']) if n_peaks > 1 else np.zeros((amps_samples.shape[0], 0))
    eta_off_diag_samples = np.array(samples['eta_off_diag']) if n_peaks > 1 else np.zeros((amps_samples.shape[0], 0))

    amps_hat = np.mean(amps_samples, axis=0)
    mus_hat = np.mean(mus_samples, axis=0)
    sigmas_hat = np.maximum(np.mean(sigmas_samples, axis=0), 1e-8)
    betas_hat = np.mean(betas_samples, axis=0)
    taus_hat = np.mean(taus_samples, axis=0)
    lambdas_hat = np.mean(lambdas_samples, axis=0)

    if n_peaks > 1:
        gamma_complex_samples = gamma_mag_samples * np.exp(1j * gamma_phase_samples)
        gamma_complex_hat = np.mean(gamma_complex_samples, axis=0)
        gamma_mag_hat = np.abs(gamma_complex_hat)
        gamma_phase_hat = np.angle(gamma_complex_hat)
        eta_off_diag_hat = np.mean(eta_off_diag_samples, axis=0)
    else:
        gamma_mag_hat = np.array([])
        gamma_phase_hat = np.array([])
        eta_off_diag_hat = np.array([])
    
    # Plot posterior diagnostics for all fitted parameters.
    posterior_panels = []

    for k in range(n_peaks):
        posterior_panels.append((f'$D_{{{k+1}{k+1}}}$', amps_samples[:, k]))
        posterior_panels.append((f'$\\mu_{k+1}$', mus_samples[:, k]))
        posterior_panels.append((f'$\\sigma_{k+1}$', sigmas_samples[:, k]))
        posterior_panels.append((f'$\\beta_{k+1}$', betas_samples[:, k]))
        posterior_panels.append((f'$\\tau_{k+1}$', taus_samples[:, k]))
        posterior_panels.append((f'$\\gamma_{k+1}$', lambdas_samples[:, k]))

    pair_idx = 0
    for i in range(n_peaks):
        for j in range(i + 1, n_peaks):
            posterior_panels.append((f'$|D_{{{i+1}{j+1}}}| / \\sqrt{{D_{{{i+1}{i+1}}} D_{{{j+1}{j+1}}}}}$', gamma_mag_samples[:, pair_idx]))
            posterior_panels.append((f'$\\text{{arg}}(D_{{{i+1}{j+1}}})$', gamma_phase_samples[:, pair_idx]))
            posterior_panels.append((f'$\\eta_{{{i+1}{j+1}}}$', eta_off_diag_samples[:, pair_idx]))
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
        ax.set_title(label, fontsize=20, y=1.04)

    for ax in axs[n_panels:]:
        ax.axis('off')

    fig.suptitle('Posterior distributions of fitted parameters', fontsize=30, y=1.02, weight='bold')
    plt.tight_layout()
    plt.savefig('MCMC_out/out1.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

    # Observed data diagnostics mapped to a full grid; unobserved entries stay at zero.
    x_grid = np.unique(np.sort(x_obs))
    y_grid = np.unique(np.sort(y_obs))
    extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]

    # Final inferred matrix from posterior-mean parameters.
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    gamma_hat = _build_hermitian_gamma(n_peaks, jnp.array(gamma_mag_hat), jnp.array(gamma_phase_hat))
    eta_hat = _build_symmetric_eta(n_peaks, jnp.array(eta_off_diag_hat))

    print(  jnp.array(amps_hat),
            jnp.array(mus_hat),
            jnp.array(sigmas_hat),
            jnp.array(betas_hat),
            jnp.array(taus_hat),
            jnp.array(lambdas_hat),
            gamma_hat,
            eta_hat)

    z_inf = rho_model(
        jnp.array(Xg),
        jnp.array(Yg),
        jnp.array(amps_hat),
        jnp.array(mus_hat),
        jnp.array(sigmas_hat),
        jnp.array(betas_hat),
        jnp.array(taus_hat),
        jnp.array(lambdas_hat),
        gamma_hat,
        eta_hat,
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

    return amps_hat, mus_hat, sigmas_hat, betas_hat, taus_hat, lambdas_hat, gamma_hat, eta_hat