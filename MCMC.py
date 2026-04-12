import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_data(size=20, true_s=2.0, true_mu=5.0, true_a=0.5, true_b=1.0, true_c=0.0, noise_level=0.1, uncertainty_scale=2.0, drop_fraction=0.3, seed=42):
    """
    Generate synthetic 2D complex matrix data.
    The phase for a pure state is phi(x) - phi(y),
    with phi(x) = a (x - mu)^2 + b (x - mu) + c.
    (Note that parameter c will cancel out in the density matrix phase).
    """
    np.random.seed(seed)
    
    # Create coordinate grid
    x_grid = np.linspace(0, 10, size)
    y_grid = np.linspace(0, 10, size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate exact amplitude
    Z_amp = np.exp(-1.0 / (2 * true_s) * ((X - true_mu)**2 + (Y - true_mu)**2))
    
    # Calculate phase (Note: global phase true_c cancels out)
    phi_X = true_a * (X - true_mu)**2 + true_b * (X - true_mu) + true_c
    phi_Y = true_a * (Y - true_mu)**2 + true_b * (Y - true_mu) + true_c
    Phi = phi_X - phi_Y
    
    # Combine amplitude and phase
    Z_exact = Z_amp * np.exp(1j * Phi)
    
    # Add amplitude-dependent noise independently to real and imaginary parts.
    sigma_obs = noise_level * (1.0 + uncertainty_scale * (1.0 - Z_amp))
    noise_real = np.random.normal(0, sigma_obs, X.shape)
    noise_imag = np.random.normal(0, sigma_obs, X.shape)
    Z_noisy = Z_exact + noise_real + 1j * noise_imag
    
    # Flatten the arrays
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z_noisy.flatten()
    sigma_flat = sigma_obs.flatten()
    
    # Remove some of the datapoints (simulate missing data)
    num_points = len(x_flat)
    num_keep = int(num_points * (1 - drop_fraction))
    
    keep_indices = np.random.choice(num_points, num_keep, replace=False)
    
    x_obs = x_flat[keep_indices]
    y_obs = y_flat[keep_indices]
    z_obs = z_flat[keep_indices]
    sigma_obs = sigma_flat[keep_indices]
    
    return x_obs, y_obs, z_obs, sigma_obs

def model(x, y, sigma_obs, z_obs=None):
    """
    Numpyro model for the 2D Gaussian density matrix with complex phase.
    """
    # Priors for the parameters
    mu = numpyro.sample('mu', dist.Uniform(0.0, 10.0))
    s = numpyro.sample('s', dist.HalfNormal(5.0))
    
    # Phase parameters
    a = numpyro.sample('a', dist.Normal(0.0, 2.0))
    b = numpyro.sample('b', dist.Normal(0.0, 2.0))
    
    # Note: we do not sample parameter 'c' because phi(x) - phi(y) eliminates 'c'. 
    # It is completely unidentifiable from this pure state matrix observable framework.

    # Expected value of the matrix amplitude
    amp = jnp.exp(-1.0 / (2 * s) * ((x - mu)**2 + (y - mu)**2))
    
    # Expected value of the phase
    phi_x = a * (x - mu)**2 + b * (x - mu)
    phi_y = a * (y - mu)**2 + b * (y - mu)
    phase = phi_x - phi_y
    
    # Real and Imaginary components
    z_real_expected = amp * jnp.cos(phase)
    z_imag_expected = amp * jnp.sin(phase)
    
    # Use provided per-observation uncertainties for both real and imaginary parts.
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs_real', dist.Normal(z_real_expected, sigma_obs), obs=jnp.real(z_obs) if z_obs is not None else None)
        numpyro.sample('obs_imag', dist.Normal(z_imag_expected, sigma_obs), obs=jnp.imag(z_obs) if z_obs is not None else None)

def main():
    print("Generating synthetic complex data...")
    # Generate data with known parameters
    x_obs, y_obs, z_obs, sigma_obs = generate_data(
        size=20, 
        true_s=2.0, 
        true_mu=5.0, 
        true_a=0.5, 
        true_b=1.0, 
        noise_level=0.3, 
        drop_fraction=0.3
    )
    print(f"Number of observations: {len(x_obs)}")

    print(type(x_obs))
    print(type(y_obs))
    print(type(z_obs))
    print(type(sigma_obs))

    
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
        z_obs=jnp.array(z_obs)
    )
    
    print("\nMCMC Summary:")
    mcmc.print_summary()
    
    # Extract posterior samples
    samples = mcmc.get_samples()
    mu_samples = samples['mu']
    s_samples = samples['s']
    a_samples = samples['a']
    b_samples = samples['b']

    mu_hat = float(np.mean(mu_samples))
    s_hat = max(float(np.mean(s_samples)), 1e-8)
    a_hat = float(np.mean(a_samples))
    b_hat = float(np.mean(b_samples))

    os.makedirs('MCMC_out', exist_ok=True)
    
    # Plot the posterior distributions
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    axs[0, 0].hist(mu_samples, bins=50, density=True, alpha=0.7, color='blue')
    axs[0, 0].axvline(5.0, color='red', linestyle='dashed', linewidth=2, label='True \\mu (5.0)')
    axs[0, 0].set_title('Posterior Distribution of \\mu')
    axs[0, 0].legend()
    
    axs[0, 1].hist(s_samples, bins=50, density=True, alpha=0.7, color='green')
    axs[0, 1].axvline(2.0, color='red', linestyle='dashed', linewidth=2, label='True s (2.0)')
    axs[0, 1].set_title('Posterior Distribution of s')
    axs[0, 1].legend()

    axs[1, 0].hist(a_samples, bins=50, density=True, alpha=0.7, color='purple')
    axs[1, 0].axvline(0.5, color='red', linestyle='dashed', linewidth=2, label='True a (0.5)')
    axs[1, 0].set_title('Posterior Distribution of chirp param (a)')
    axs[1, 0].legend()

    axs[1, 1].hist(b_samples, bins=50, density=True, alpha=0.7, color='orange')
    axs[1, 1].axvline(1.0, color='red', linestyle='dashed', linewidth=2, label='True b (1.0)')
    axs[1, 1].set_title('Posterior Distribution of linear param (b)')
    axs[1, 1].legend()
    
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

    amp_inf = np.exp(-1.0 / (2.0 * s_hat) * ((Xg - mu_hat) ** 2 + (Yg - mu_hat) ** 2))
    phase_inf = (a_hat * (Xg - mu_hat) ** 2 + b_hat * (Xg - mu_hat)) - (
        a_hat * (Yg - mu_hat) ** 2 + b_hat * (Yg - mu_hat)
    )

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

if __name__ == "__main__":
    main()
