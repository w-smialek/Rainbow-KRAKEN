import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
hbar = 6.582119569e-1


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


def _fftconvolve_same_last_axis_precomputed_left_fft(a_fft, b, n_a):
    """Same-mode convolution when FFT(a) is precomputed along the last axis."""
    n_b = b.shape[-1]
    n_full = n_a + n_b - 1

    fb = jnp.fft.fft(b, n=n_full, axis=-1)
    full = jnp.fft.ifft(a_fft * fb, axis=-1)

    start = (n_b - 1) // 2
    end = start + n_a
    return full[..., start:end]


def probe_magnitude_n_gaussians(omega, amps, means, sigmas):
    """Real non-negative probe magnitude modeled as a sum of Gaussians."""
    omega = jnp.asarray(omega)
    amps = jnp.asarray(amps)
    means = jnp.asarray(means)
    sigmas = jnp.clip(jnp.asarray(sigmas), 1e-12)

    ndim = omega.ndim
    pshape = (slice(None),) + (None,) * ndim
    omega_e = omega[None, ...]
    amps_e = amps[pshape]
    means_e = means[pshape]
    sigmas_e = sigmas[pshape]

    gauss = amps_e * jnp.exp(-0.5 * ((omega_e - means_e) / sigmas_e) ** 2)
    return jnp.sum(gauss, axis=0)


def probe_global_phase(omega, om_ref, phase_grad, chirp):
    """Global phase model with constant removed: phi = grad*domega + 0.5*chirp*domega^2."""
    domega = omega - om_ref
    return phase_grad * domega + 0.5 * chirp * domega ** 2


def probe_field_n_gaussians(omega, om_ref, amps, means, sigmas, phase_grad, chirp):
    mag = probe_magnitude_n_gaussians(omega, amps, means, sigmas)
    ph = probe_global_phase(omega, om_ref, phase_grad, chirp)
    return mag * jnp.exp(1j * ph)


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


def _prepare_zero_omega_context(om_p, om_x, om_t, rho_params, om_ref):
    """Precompute forward-model terms independent of probe parameters."""
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

    n_energy = in_1.shape[-1]
    n_full = 2 * n_energy - 1

    return {
        "in_1_fft": jnp.fft.fft(in_1, n=n_full, axis=-1),
        "inv_denom": 1.0 / _safe_real_denom((om_p + om_t) * om_p),
        "d_om": om_p[0, 1] - om_p[0, 0],
    }


def zero_omega_forward_model(
    om_p,
    om_x,
    om_t,
    rho_params,
    om_ref,
    probe_amps,
    probe_means,
    probe_sigmas,
    probe_phase_grad,
    probe_chirp,
    zero_ctx=None,
):
    """
    Forward model for the zero-omega component used in WF_pipeline_zero.

    Implements:
      in_1 = rho((OM_X - OM_T + om_ref)*hbar, (OM_X + om_ref)*hbar)
      in_2 = sp(OM_P + OM_T) * conj(sp(OM_P)) / ((OM_P + OM_T)*OM_P)
      zerocomp = fftconvolve(in_1, in_2, mode='same', axis=energy) * d_om
    """
    if zero_ctx is None:
        zero_ctx = _prepare_zero_omega_context(om_p, om_x, om_t, rho_params, om_ref)

    sp_shift = probe_field_n_gaussians(
        om_p + om_t,
        om_ref,
        probe_amps,
        probe_means,
        probe_sigmas,
        probe_phase_grad,
        probe_chirp,
    )
    sp_now = probe_field_n_gaussians(
        om_p,
        om_ref,
        probe_amps,
        probe_means,
        probe_sigmas,
        probe_phase_grad,
        probe_chirp,
    )

    in_2 = sp_shift * jnp.conj(sp_now) * zero_ctx["inv_denom"]

    n_energy = in_2.shape[-1]
    zerocomp = _fftconvolve_same_last_axis_precomputed_left_fft(
        zero_ctx["in_1_fft"],
        in_2,
        n_energy,
    ) * zero_ctx["d_om"]
    return zerocomp


def model(
    om_p,
    om_x,
    om_t,
    sigma_obs,
    rho_params,
    om_ref,
    n_probe_peaks,
    z_obs=None,
    obs_mask=None,
    zero_ctx=None,
):
    """NumPyro model with N-Gaussian probe magnitude plus global phase gradient/chirp."""
    om_min = jnp.min(om_p)
    om_max = jnp.max(om_p)

    probe_amps_raw = numpyro.sample(
        "probe_amps_raw",
        dist.HalfNormal(200.00).expand([n_probe_peaks]).to_event(1),
    )
    probe_means_raw = numpyro.sample(
        "probe_means_raw",
        dist.Uniform(1.50, 4.50).expand([n_probe_peaks]).to_event(1),
    )
    probe_sigmas_raw = numpyro.sample(
        "probe_sigmas_raw",
        dist.HalfNormal(0.15)
        .expand([n_probe_peaks])
        .to_event(1),
    )
    probe_phase_grad = numpyro.sample(
        "probe_phase_grad",
        dist.Normal(0.0, 0.1),
    )
    probe_chirp = numpyro.sample(
        "probe_chirp",
        dist.Normal(0.0, 0.5),
    )

    # Sort by center frequency in the forward model to reduce label-switching ambiguity.
    sort_idx = jnp.argsort(probe_means_raw)
    probe_means = probe_means_raw[sort_idx]
    probe_amps = probe_amps_raw[sort_idx]
    probe_sigmas = probe_sigmas_raw[sort_idx]

    z_expected = zero_omega_forward_model(
        om_p,
        om_x,
        om_t,
        rho_params,
        om_ref,
        probe_amps,
        probe_means,
        probe_sigmas,
        probe_phase_grad,
        probe_chirp,
        zero_ctx=zero_ctx,
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
    n_probe_peaks=1,
    obs_mask=None,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    rng_seed=0,
    save_prefix="MCMC_out/probe",
):
    """
        Infer probe parameters from a zero-omega signal component.

        Probe model:
            |S_probe(omega)| = sum_k A_k * exp(-0.5 * ((omega - mu_k)/sigma_k)^2)
            phase(omega) = phase_grad * (omega - om_ref) + 0.5 * chirp * (omega - om_ref)^2

        A constant phase offset is intentionally omitted (irrelevant for this model).

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
    probe_amps_hat, probe_means_hat, probe_sigmas_hat, probe_phase_grad_hat, probe_chirp_hat, z_fit
    """
    om_p = jnp.asarray(om_p)
    om_x = jnp.asarray(om_x)
    om_t = jnp.asarray(om_t)
    z_obs = jnp.asarray(z_obs)
    sigma_obs = jnp.asarray(sigma_obs)

    zero_ctx = _prepare_zero_omega_context(om_p, om_x, om_t, rho_params, om_ref)

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
        n_probe_peaks=n_probe_peaks,
        z_obs=z_obs,
        obs_mask=obs_mask,
        zero_ctx=zero_ctx,
    )

    mcmc.print_summary()
    samples = mcmc.get_samples()

    probe_amps_raw_samples = np.asarray(samples["probe_amps_raw"])
    probe_means_raw_samples = np.asarray(samples["probe_means_raw"])
    probe_sigmas_raw_samples = np.asarray(samples["probe_sigmas_raw"])
    probe_phase_grad_samples = np.asarray(samples["probe_phase_grad"])
    probe_chirp_samples = np.asarray(samples["probe_chirp"])

    # Sort each posterior draw by center frequency for consistent component indexing.
    sort_idx = np.argsort(probe_means_raw_samples, axis=1)
    probe_means_samples = np.take_along_axis(probe_means_raw_samples, sort_idx, axis=1)
    probe_amps_samples = np.take_along_axis(probe_amps_raw_samples, sort_idx, axis=1)
    probe_sigmas_samples = np.take_along_axis(probe_sigmas_raw_samples, sort_idx, axis=1)

    probe_amps_hat = np.mean(probe_amps_samples, axis=0)
    probe_means_hat = np.mean(probe_means_samples, axis=0)
    probe_sigmas_hat = np.mean(probe_sigmas_samples, axis=0)
    probe_phase_grad_hat = float(np.mean(probe_phase_grad_samples))
    probe_chirp_hat = float(np.mean(probe_chirp_samples))

    z_fit = zero_omega_forward_model(
        om_p,
        om_x,
        om_t,
        rho_params,
        om_ref,
        probe_amps_hat,
        probe_means_hat,
        probe_sigmas_hat,
        probe_phase_grad_hat,
        probe_chirp_hat,
        zero_ctx=zero_ctx,
    )

    posterior_panels = []
    for k in range(n_probe_peaks):
        posterior_panels.append((f"Probe amp {k+1}", probe_amps_samples[:, k]))
        posterior_panels.append((f"Probe mean {k+1}", probe_means_samples[:, k]))
        posterior_panels.append((f"Probe sigma {k+1}", probe_sigmas_samples[:, k]))
    posterior_panels.append(("Phase gradient", probe_phase_grad_samples))
    posterior_panels.append(("Chirp", probe_chirp_samples))

    n_panels = len(posterior_panels)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.8 * n_rows))
    axs = np.atleast_1d(axs).ravel()

    for ax, (label, values) in zip(axs, posterior_panels):
        ax.hist(values, bins=40, density=True, alpha=0.8, color="tab:blue")
        ax.axvline(np.mean(values), color="black", linestyle="--", linewidth=1.0)
        ax.set_title(label)
    for ax in axs[n_panels:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_posterior.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return (
        np.asarray(probe_amps_hat),
        np.asarray(probe_means_hat),
        np.asarray(probe_sigmas_hat),
        probe_phase_grad_hat,
        probe_chirp_hat,
        np.asarray(z_fit),
    )
