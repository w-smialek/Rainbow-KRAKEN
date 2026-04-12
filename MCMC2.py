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

# def generate_data(size=20, true_s=2.0, true_mu=5.0, true_a=0.5, true_b=1.0, true_c=0.0, noise_level=0.1, uncertainty_scale=2.0, drop_fraction=0.3, seed=42):
#     """
#     Generate synthetic 2D complex matrix data.
#     The phase for a pure state is phi(x) - phi(y),
#     with phi(x) = a (x - mu)^2 + b (x - mu) + c.
#     (Note that parameter c will cancel out in the density matrix phase).
#     """
#     np.random.seed(seed)
    
#     # Create coordinate grid
#     x_grid = np.linspace(0, 10, size)
#     y_grid = np.linspace(0, 10, size)
#     X, Y = np.meshgrid(x_grid, y_grid)
    
#     # Calculate exact amplitude
#     Z_amp = np.exp(-1.0 / (2 * true_s) * ((X - true_mu)**2 + (Y - true_mu)**2))
    
#     # Calculate phase (Note: global phase true_c cancels out)
#     phi_X = true_a * (X - true_mu)**2 + true_b * (X - true_mu) + true_c
#     phi_Y = true_a * (Y - true_mu)**2 + true_b * (Y - true_mu) + true_c
#     Phi = phi_X - phi_Y
    
#     # Combine amplitude and phase
#     Z_exact = Z_amp * np.exp(1j * Phi)
    
#     # Add amplitude-dependent noise independently to real and imaginary parts.
#     sigma_obs = noise_level * (1.0 + uncertainty_scale * (1.0 - Z_amp))
#     noise_real = np.random.normal(0, sigma_obs, X.shape)
#     noise_imag = np.random.normal(0, sigma_obs, X.shape)
#     Z_noisy = Z_exact + noise_real + 1j * noise_imag
    
#     # Flatten the arrays
#     x_flat = X.flatten()
#     y_flat = Y.flatten()
#     z_flat = Z_noisy.flatten()
#     sigma_flat = sigma_obs.flatten()
    
#     # Remove some of the datapoints (simulate missing data)
#     num_points = len(x_flat)
#     num_keep = int(num_points * (1 - drop_fraction))
    
#     keep_indices = np.random.choice(num_points, num_keep, replace=False)
    
#     x_obs = x_flat[keep_indices]
#     y_obs = y_flat[keep_indices]
#     z_obs = z_flat[keep_indices]
#     sigma_obs = sigma_flat[keep_indices]
    
#     return x_obs, y_obs, z_obs, sigma_obs

def rho_peak(x, amp, mu, sigma, a, b, c):
    return amp / (2 * sigma) * jnp.exp(
        -(x - mu) ** 2 / (2 * sigma ** 2) + 1j * (a * (x - mu) ** 2 + b * (x - mu) + c)
    )

def rho_model(e1, e2, amps, mus, sigmas, a_s, bs, cs, Lambdas, gammas):
    retval = jnp.zeros_like(e1, dtype=jnp.complex128)
    for k in range(len(amps)):
        for l in range(len(amps)):
            retval += (
                gammas[k, l]
                * rho_peak(e1, amps[k], mus[k], sigmas[k], a_s[k], bs[k], cs[k])
                * jnp.conj(rho_peak(e2, amps[l], mus[l], sigmas[l], a_s[l], bs[l], cs[l]))
                * jnp.exp(-0.5 * (Lambdas[k] + Lambdas[l]) * (e1 - e2) ** 2)
            )
    return retval


def _build_hermitian_gamma(n_peaks, gamma_mag, gamma_phase):
    gamma = jnp.eye(n_peaks, dtype=jnp.complex128)
    idx = 0
    for i in range(n_peaks):
        for j in range(i + 1, n_peaks):
            gij = gamma_mag[idx] * jnp.exp(1j * gamma_phase[idx])
            gamma = gamma.at[i, j].set(gij)
            gamma = gamma.at[j, i].set(jnp.conj(gij))
            idx += 1
    return gamma

om_ref = 1.55 / hbar
amps = [1.0 / np.sqrt(2), 1.0]
mus = [25.0 - 0.17 + om_ref * hbar, 25.0 + om_ref * hbar]
sigmas = [0.1, 0.1]
a_s = [0, 4]
bs = [3, 0]
cs = [0, 0]
Lambdas = [10, 10]
gammas = np.array([[1.0, 0.0],
                   [0.0, 1.0]])

def model(x, y, sigma_obs, z_obs=None, n_peaks=2):
    """
    Numpyro model for the 2D Gaussian density matrix with complex phase.
    """
    # Priors per peak (same prior family for each component).
    amps_now = numpyro.sample(
        'amps', dist.Uniform(0.0, 20000.0).expand([n_peaks]).to_event(1)
    )
    mus_now = numpyro.sample(
        'mus', dist.Uniform(24.0, 26.0).expand([n_peaks]).to_event(1)
    )
    sigmas_now = numpyro.sample(
        'sigmas', dist.HalfNormal(0.5).expand([n_peaks]).to_event(1)
    )
    a_now = numpyro.sample(
        'a_s', dist.Normal(0.0, 8.0).expand([n_peaks]).to_event(1)
    )
    b_now = numpyro.sample(
        'b_s', dist.Normal(0.0, 8.0).expand([n_peaks]).to_event(1)
    )
    lambdas_now = numpyro.sample(
        'Lambdas', dist.HalfNormal(10.0).expand([n_peaks]).to_event(1)
    )

    # Peak phases with gauge fixing: global phase is unidentifiable, so c_0 = 0.
    if n_peaks > 1:
        c_free = numpyro.sample(
            'c_free', dist.Uniform(-jnp.pi, jnp.pi).expand([n_peaks - 1]).to_event(1)
        )
        cs_now = jnp.concatenate([jnp.array([0.0]), c_free])
    else:
        cs_now = jnp.array([0.0])

    # Complex Hermitian gamma with unit diagonal and free off-diagonal entries.
    n_pairs = n_peaks * (n_peaks - 1) // 2
    if n_pairs > 0:
        gamma_mag = numpyro.sample(
            'gamma_mag', dist.Uniform(0.0, 1.0).expand([n_pairs]).to_event(1)
        )
        gamma_phase = numpyro.sample(
            'gamma_phase', dist.Uniform(-jnp.pi, jnp.pi).expand([n_pairs]).to_event(1)
        )
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
        cs_now,
        lambdas_now,
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
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000, num_chains=1)
    
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
    a_samples = np.array(samples['a_s'])
    b_samples = np.array(samples['b_s'])
    lambdas_samples = np.array(samples['Lambdas'])
    c_free_samples = np.array(samples['c_free']) if n_peaks > 1 else np.zeros((amps_samples.shape[0], 0))
    gamma_mag_samples = np.array(samples['gamma_mag']) if n_peaks > 1 else np.zeros((amps_samples.shape[0], 0))
    gamma_phase_samples = np.array(samples['gamma_phase']) if n_peaks > 1 else np.zeros((amps_samples.shape[0], 0))

    amps_hat = np.mean(amps_samples, axis=0)
    mus_hat = np.mean(mus_samples, axis=0)
    sigmas_hat = np.maximum(np.mean(sigmas_samples, axis=0), 1e-8)
    a_hat = np.mean(a_samples, axis=0)
    b_hat = np.mean(b_samples, axis=0)
    lambdas_hat = np.maximum(np.mean(lambdas_samples, axis=0), 1e-8)

    if n_peaks > 1:
        cs_hat = np.concatenate(([0.0], _circular_mean(c_free_samples, axis=0)))
        gamma_complex_samples = gamma_mag_samples * np.exp(1j * gamma_phase_samples)
        gamma_complex_hat = np.mean(gamma_complex_samples, axis=0)
        gamma_mag_hat = np.abs(gamma_complex_hat)
        gamma_phase_hat = np.angle(gamma_complex_hat)
    else:
        cs_hat = np.array([0.0])
        gamma_mag_hat = np.array([])
        gamma_phase_hat = np.array([])

    os.makedirs('MCMC_out', exist_ok=True)
    
    # Plot compact posterior diagnostics for key per-peak parameters.
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for k in range(n_peaks):
        axs[0, 0].hist(mus_samples[:, k], bins=40, density=True, alpha=0.35, label=f'peak {k}')
        axs[0, 1].hist(sigmas_samples[:, k], bins=40, density=True, alpha=0.35, label=f'peak {k}')
        axs[1, 0].hist(a_samples[:, k], bins=40, density=True, alpha=0.35, label=f'peak {k}')
        axs[1, 1].hist(b_samples[:, k], bins=40, density=True, alpha=0.35, label=f'peak {k}')

    axs[0, 0].set_title('Posterior of mus')
    axs[0, 1].set_title('Posterior of sigmas')
    axs[1, 0].set_title('Posterior of a_s')
    axs[1, 1].set_title('Posterior of b_s')
    axs[0, 0].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('MCMC_out/out1.png')
    plt.close(fig)

    # Observed data diagnostics: amplitude, phase, and known per-point uncertainty.
    z_amp_obs = np.abs(z_obs)
    z_phase_obs = np.angle(z_obs)

    fig_obs, axs_obs = plt.subplots(1, 3, figsize=(18, 5))

    sc_amp = axs_obs[0].scatter(x_obs, y_obs, c=z_amp_obs, s=30, cmap='viridis')
    axs_obs[0].set_title('Observed |z_obs|')
    axs_obs[0].set_xlabel('x_obs')
    axs_obs[0].set_ylabel('y_obs')
    axs_obs[0].set_aspect('equal', adjustable='box')
    fig_obs.colorbar(sc_amp, ax=axs_obs[0], label='Amplitude')

    sc_phase = axs_obs[1].scatter(x_obs, y_obs, c=z_phase_obs, s=30, cmap='twilight')
    axs_obs[1].set_title('Observed phase(z_obs)')
    axs_obs[1].set_xlabel('x_obs')
    axs_obs[1].set_ylabel('y_obs')
    axs_obs[1].set_aspect('equal', adjustable='box')
    fig_obs.colorbar(sc_phase, ax=axs_obs[1], label='Phase [rad]')

    sc_sigma = axs_obs[2].scatter(x_obs, y_obs, c=sigma_obs, s=30, cmap='magma')
    axs_obs[2].set_title('Provided sigma_obs')
    axs_obs[2].set_xlabel('x_obs')
    axs_obs[2].set_ylabel('y_obs')
    axs_obs[2].set_aspect('equal', adjustable='box')
    fig_obs.colorbar(sc_sigma, ax=axs_obs[2], label='Sigma')

    plt.tight_layout()
    plt.savefig('MCMC_out/diagnostic_observed.png')
    plt.close(fig_obs)

    # Final inferred matrix from posterior-mean parameters.
    x_grid = np.unique(np.sort(x_obs))
    y_grid = np.unique(np.sort(y_obs))
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    gamma_hat = _build_hermitian_gamma(n_peaks, jnp.array(gamma_mag_hat), jnp.array(gamma_phase_hat))

    z_inf = rho_model(
        jnp.array(Xg),
        jnp.array(Yg),
        jnp.array(amps_hat),
        jnp.array(mus_hat),
        jnp.array(sigmas_hat),
        jnp.array(a_hat),
        jnp.array(b_hat),
        jnp.array(cs_hat),
        jnp.array(lambdas_hat),
        gamma_hat,
    )
    amp_inf = np.abs(np.array(z_inf))
    phase_inf = np.angle(np.array(z_inf))

    fig_inf, axs_inf = plt.subplots(1, 2, figsize=(12, 5))
    extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]

    im_amp = axs_inf[0].imshow(amp_inf, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    axs_inf[0].set_title('Inferred matrix amplitude')
    axs_inf[0].set_xlabel('x')
    axs_inf[0].set_ylabel('y')
    fig_inf.colorbar(im_amp, ax=axs_inf[0], label='Amplitude')

    im_phase = axs_inf[1].imshow(phase_inf, origin='lower', extent=extent, aspect='auto', cmap='twilight')
    axs_inf[1].set_title('Inferred matrix phase')
    axs_inf[1].set_xlabel('x')
    axs_inf[1].set_ylabel('y')
    fig_inf.colorbar(im_phase, ax=axs_inf[1], label='Phase [rad]')

    plt.tight_layout()
    plt.savefig('MCMC_out/inferred_matrix.png')
    plt.close(fig_inf)