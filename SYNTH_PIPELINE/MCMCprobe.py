import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt


def rho_model(e1, e2, amps, mus, sigmas, betas, taus, lambdas, gammas, etas):
    """Density-matrix model used as a known constant in probe-parameter inference."""
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

    rho1 = amps_k / (2.0 * jnp.pi * sigmas_k ** 2) ** 0.25 * jnp.exp(
        -(x1 ** 2) * (1.0 / (4.0 * sigmas_k ** 2) - 1j * beta_k / 2.0)
        + 1j * tau_k * x1
    )
    rho2 = amps_k / (2.0 * jnp.pi * sigmas_k ** 2) ** 0.25 * jnp.exp(
        -(x2 ** 2) * (1.0 / (4.0 * sigmas_k ** 2) - 1j * beta_k / 2.0)
        + 1j * tau_k * x2
    )

    gamma_e = gammas[(slice(None), slice(None)) + (None,) * ndim]
    eta_e = etas[(slice(None), slice(None)) + (None,) * ndim]

    d_kl = jnp.exp(
        -0.5 * lambda_k[:, None, ...] ** 2 * x1[:, None, ...] ** 2
        -0.5 * lambda_k[None, :, ...] ** 2 * x2[None, :, ...] ** 2
        + eta_e
        * lambda_k[:, None, ...]
        * lambda_k[None, :, ...]
        * x1[:, None, ...]
        * x2[None, :, ...]
    )

    pair_terms = gamma_e * rho1[:, None, ...] * jnp.conj(rho2)[None, :, ...] * d_kl
    return jnp.sum(pair_terms, axis=(0, 1))


def _safe_real_denom(x, eps=1e-12):
    sign = jnp.where(x < 0.0, -1.0, 1.0)
    return jnp.where(jnp.abs(x) < eps, sign * eps, x)


def _fftconvolve_same_last_axis(a, b):
    """Linear convolution with SciPy-like mode='same' along the last axis."""
    n_a = a.shape[-1]
    n_b = b.shape[-1]
    n_full = n_a + n_b - 1

    fa = jnp.fft.fft(a, n=n_full, axis=-1)
    fb = jnp.fft.fft(b, n=n_full, axis=-1)
    full = jnp.fft.ifft(fa * fb, axis=-1)

    start = (n_b - 1) // 2
    end = start + n_a
    return full[..., start:end]


def probe_spectrum_single_gaussian(omega, amp, mean, sigma):
    sigma_eff = jnp.clip(sigma, 1e-12)
    return amp * jnp.exp(-0.5 * ((omega - mean) / sigma_eff) ** 2)


def _unpack_rho_params(rho_params):
    required = [
        "amps",
        "mus",
        "sigmas",
        "betas",
        "taus",
        "lambdas",
        "gammas",
        "etas",
    ]
    missing = [k for k in required if k not in rho_params]
    if missing:
        missing_txt = ", ".join(missing)
        raise ValueError(f"Missing rho_params entries: {missing_txt}")

    return {k: jnp.asarray(rho_params[k]) for k in required}


def zero_omega_forward_model(
    om_p,
    om_x,
    om_t,
    rho_params,
    om_ref,
    hbar,
    probe_amp,
    probe_mean,
    probe_sigma,
):
    """
    Forward model for the zero-omega component used in WF_pipeline_zero.

    Implements:
      in_1 = rho((OM_X - OM_T + om_ref)*hbar, (OM_X + om_ref)*hbar)
      in_2 = sp(OM_P + OM_T) * conj(sp(OM_P)) / ((OM_P + OM_T)*OM_P)
      zerocomp = fftconvolve(in_1, in_2, mode='same', axis=energy) * d_om
    """
    rho = _unpack_rho_params(rho_params)

    in_1 = rho_model(
        (om_x - om_t + om_ref) * hbar,
        (om_x + om_ref) * hbar,
        rho["amps"],
        rho["mus"],
        rho["sigmas"],
        rho["betas"],
        rho["taus"],
        rho["lambdas"],
        rho["gammas"],
        rho["etas"],
    )

    sp_shift = probe_spectrum_single_gaussian(om_p + om_t, probe_amp, probe_mean, probe_sigma)
    sp_now = probe_spectrum_single_gaussian(om_p, probe_amp, probe_mean, probe_sigma)

    denom = _safe_real_denom((om_p + om_t) * om_p)
    in_2 = sp_shift * jnp.conj(sp_now) / denom

    d_om = om_p[0, 1] - om_p[0, 0]
    zerocomp = _fftconvolve_same_last_axis(in_1, in_2) * d_om
    return zerocomp


def model(
    om_p,
    om_x,
    om_t,
    sigma_obs,
    rho_params,
    om_ref,
    hbar,
    amp_prior_scale,
    sigma_prior_center,
    z_obs=None,
    obs_mask=None,
):
    """NumPyro model that samples only probe (amp, mean, sigma) parameters."""
    om_min = jnp.min(om_p)
    om_max = jnp.max(om_p)

    probe_amp = numpyro.sample("probe_amp", dist.HalfNormal(jnp.clip(amp_prior_scale, 1e-12)))
    probe_mean = numpyro.sample("probe_mean", dist.Uniform(om_min, om_max))
    probe_sigma = numpyro.sample(
        "probe_sigma",
        dist.LogNormal(jnp.log(jnp.clip(sigma_prior_center, 1e-12)), 0.8),
    )

    z_expected = zero_omega_forward_model(
        om_p,
        om_x,
        om_t,
        rho_params,
        om_ref,
        hbar,
        probe_amp,
        probe_mean,
        probe_sigma,
    )

    sigma_eff = jnp.clip(jnp.asarray(sigma_obs), 1e-12)

    z_real = jnp.real(z_expected).reshape(-1)
    z_imag = jnp.imag(z_expected).reshape(-1)
    sig = sigma_eff.reshape(-1)

    if z_obs is None:
        obs_real = None
        obs_imag = None
    else:
        z_obs = jnp.asarray(z_obs)
        obs_real = jnp.real(z_obs).reshape(-1)
        obs_imag = jnp.imag(z_obs).reshape(-1)

    if obs_mask is None:
        mask = jnp.ones_like(z_real, dtype=bool)
    else:
        mask = jnp.asarray(obs_mask, dtype=bool).reshape(-1)

    with numpyro.plate("data", z_real.shape[0]):
        numpyro.sample("obs_real", dist.Normal(z_real, sig).mask(mask), obs=obs_real)
        numpyro.sample("obs_imag", dist.Normal(z_imag, sig).mask(mask), obs=obs_imag)


def Bayesian_MCMC(
    om_p,
    om_x,
    om_t,
    z_obs,
    sigma_obs,
    rho_params,
    om_ref,
    hbar=6.582119569e-1,
    obs_mask=None,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    rng_seed=0,
    amp_prior_scale=None,
    sigma_prior_center=None,
    save_prefix="MCMC_out/probe",
):
    """
    Infer single-Gaussian probe parameters from a zero-omega signal component.

    Parameters
    ----------
    om_p, om_x, om_t : array-like, shape (N_t, N_e)
        2D frequency grids used in the zero-omega forward model.
    z_obs : array-like, shape (N_t, N_e), complex
        Observed complex zero-omega component.
    sigma_obs : array-like, shape (N_t, N_e) or scalar
        Observation uncertainty for both real and imaginary parts.
    rho_params : dict
        Known density-matrix parameters with keys:
        amps, mus, sigmas, betas, taus, lambdas, gammas, etas.
    om_ref : float
        Reference frequency used in the forward model.

    Returns
    -------
    probe_amp_hat, probe_mean_hat, probe_sigma_hat, z_fit
    """
    om_p = jnp.asarray(om_p)
    om_x = jnp.asarray(om_x)
    om_t = jnp.asarray(om_t)
    z_obs = jnp.asarray(z_obs)

    if om_p.shape != om_x.shape or om_p.shape != om_t.shape:
        raise ValueError("om_p, om_x, and om_t must have the same 2D shape")

    if z_obs.shape != om_p.shape:
        raise ValueError("z_obs must have the same shape as om_p")

    sigma_obs = jnp.asarray(sigma_obs)
    if sigma_obs.ndim == 0:
        sigma_obs = jnp.full(om_p.shape, sigma_obs)
    if sigma_obs.shape != om_p.shape:
        raise ValueError("sigma_obs must be scalar or have the same shape as om_p")

    om_span = float(np.max(np.asarray(om_p)) - np.min(np.asarray(om_p)))
    if amp_prior_scale is None:
        amp_prior_scale = float(np.std(np.abs(np.asarray(z_obs))))
        amp_prior_scale = max(amp_prior_scale, 1e-6)
    if sigma_prior_center is None:
        sigma_prior_center = max(om_span / 20.0, 1e-6)

    nuts_kernel = NUTS(model, target_accept_prob=0.85)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    rng_key = jax.random.PRNGKey(rng_seed)
    mcmc.run(
        rng_key,
        om_p=om_p,
        om_x=om_x,
        om_t=om_t,
        sigma_obs=sigma_obs,
        rho_params=rho_params,
        om_ref=om_ref,
        hbar=hbar,
        amp_prior_scale=amp_prior_scale,
        sigma_prior_center=sigma_prior_center,
        z_obs=z_obs,
        obs_mask=obs_mask,
    )

    mcmc.print_summary()
    samples = mcmc.get_samples()

    probe_amp_samples = np.asarray(samples["probe_amp"])
    probe_mean_samples = np.asarray(samples["probe_mean"])
    probe_sigma_samples = np.asarray(samples["probe_sigma"])

    probe_amp_hat = float(np.mean(probe_amp_samples))
    probe_mean_hat = float(np.mean(probe_mean_samples))
    probe_sigma_hat = float(np.mean(probe_sigma_samples))

    z_fit = zero_omega_forward_model(
        om_p,
        om_x,
        om_t,
        rho_params,
        om_ref,
        hbar,
        probe_amp_hat,
        probe_mean_hat,
        probe_sigma_hat,
    )

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    axs[0].hist(probe_amp_samples, bins=40, density=True, alpha=0.8, color="tab:blue")
    axs[0].axvline(probe_amp_hat, color="black", linestyle="--", linewidth=1.0)
    axs[0].set_title("Probe amplitude")

    axs[1].hist(probe_mean_samples, bins=40, density=True, alpha=0.8, color="tab:orange")
    axs[1].axvline(probe_mean_hat, color="black", linestyle="--", linewidth=1.0)
    axs[1].set_title("Probe mean")

    axs[2].hist(probe_sigma_samples, bins=40, density=True, alpha=0.8, color="tab:green")
    axs[2].axvline(probe_sigma_hat, color="black", linestyle="--", linewidth=1.0)
    axs[2].set_title("Probe width")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_posterior.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return probe_amp_hat, probe_mean_hat, probe_sigma_hat, np.asarray(z_fit)
