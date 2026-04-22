import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt
import numpy as np

hbar = 6.582119569e-1

####
####
####


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

def _fftconvolve_same_last_axis_precomputed_left_fft(a_fft, b, n_a):
    """Same-mode convolution when FFT(a) is precomputed along the last axis."""
    n_b = b.shape[-1]
    n_full = n_a + n_b - 1

    fb = jnp.fft.fft(b, n=n_full, axis=-1)
    full = jnp.fft.ifft(a_fft * fb, axis=-1)

    start = (n_b - 1) // 2
    end = start + n_a
    return full[..., start:end]

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

def _safe_real_denom(x, eps=1e-12):
    sign = jnp.where(x < 0.0, -1.0, 1.0)
    return jnp.where(jnp.abs(x) < eps, sign * eps, x)

####
####
####

def get_f_interp(om_eval, om_grid, f_real, f_imag):
    """Interpolate f_real and f_imag evaluated at om_eval from om_grid."""
    f_r = jnp.interp(om_eval, om_grid, f_real, left=0., right=0.)
    f_i = jnp.interp(om_eval, om_grid, f_imag, left=0., right=0.)
    return f_r + 1j * f_i

def forward_model_discrete(f_real, f_imag, om_grid, om_p, om_t, zero_ctx):
    """Forward model tailored for a discrete probe vector f via interpolation."""
    om_p_plus_t = om_p + om_t
    
    sp_shift = get_f_interp(om_p_plus_t, om_grid, f_real, f_imag)
    sp_now = get_f_interp(om_p, om_grid, f_real, f_imag)
    
    in_2 = sp_shift * jnp.conj(sp_now) * zero_ctx["inv_denom"]
    n_energy = in_2.shape[-1]
    
    zerocomp = _fftconvolve_same_last_axis_precomputed_left_fft(
        zero_ctx["in_1_fft"],
        in_2,
        n_energy,
    ) * zero_ctx["d_om"]
    
    return zerocomp

def build_loss(om_grid, om_p, om_t, z_obs, sigma_obs, mask, zero_ctx, lambda1=0.0, lambda2=0.0):
    d_om = om_grid[1] - om_grid[0]
    
    @jax.jit
    def loss_fn(params):
        f_real, f_imag = params
        z_expected = forward_model_discrete(f_real, f_imag, om_grid, om_p, om_t, zero_ctx)
        
        diff = z_expected - z_obs
        sig = jnp.clip(sigma_obs, 1e-12)
        weighted_diff = diff / sig
        
        sq_err = jnp.abs(weighted_diff)**2
        
        # Heteroscedastic gaussian noise depending only on x implies dividing squared differences by sig^2
        data_term = 0.5 * jnp.sum(sq_err * mask)
        
        # Tikhonov regularization (L2)
        reg_tikhonov = jnp.sum(f_real**2 + f_imag**2) * d_om
        
        # Sobolev H1 regularization (L2 of derivative)
        df_real = jnp.diff(f_real) / d_om
        df_imag = jnp.diff(f_imag) / d_om
        reg_sobolev = jnp.sum(df_real**2 + df_imag**2) * d_om
        
        return data_term + lambda1 * reg_tikhonov + lambda2 * reg_sobolev
        
    return loss_fn

def spectral_initialization(om_grid, om_p, om_x, om_t, rho_params, om_ref, z_obs, sigma_obs, mask, n_iter=50):
    '''
    Constructs an implicitly matrix-free surrogate matrix Y matching user's initialization.
    Y.v(p) = sum_{x,t} [ S(x,p)/sigma^2(x) * rho(x-t+ref, x+ref) / ((p+t)p) ] * v(p+t)
    Finds leading eigenvector through power iteration. 
    (Note: mapping the user's notation exactly:
        omega -> p (probe freq)
        x -> om_x
        y -> om_t
        t -> omega -> p
    )
    '''
    rho = _unpack_rho_params(rho_params)
    
    # ensure grids are 1D where needed:
    om_p_1d = om_grid
    
    # Safely extract 1D domains
    om_x_1d = om_x[:, 0] if om_x.ndim > 1 else om_x
    om_t_1d = jnp.squeeze(om_t)
    
    om_x_v = om_x_1d[:, None]
    om_t_v = om_t_1d[None, :]
    
    # Pre-evaluate rho
    in_1_rho = rho_model(
        (om_x_v - om_t_v + om_ref) * hbar,
        (om_x_v + om_ref) * hbar,
        rho["amps"],
        rho["mus"],
        rho["sigmas"],
        rho["betas"],
        rho["taus"],
        rho["lambdas"],
        rho["gammas"],
        rho["etas"],
    )
    
    T = len(om_p_1d)
    
    @jax.jit
    def apply_Y(v):
        def compute_element(p_idx, p_val):
            # v_shifted evaluated at (p + t)
            v_val_interp = jnp.interp(p_val + om_t_v, om_p_1d, jnp.real(v), left=0., right=0.) + \
                           1j * jnp.interp(p_val + om_t_v, om_p_1d, jnp.imag(v), left=0., right=0.)
            
            S_x = z_obs[:, p_idx]
            sig2_x = sigma_obs[:, p_idx]**2
            mask_x = mask[:, p_idx]
            
            S_weighted = jnp.where(mask_x, S_x / jnp.clip(sig2_x, 1e-12), 0.0)
            
            denom = (p_val + om_t_v) * p_val
            safe_denom = jnp.where(denom == 0, 1e-8, denom)
            
            integrand = S_weighted[:, None] * in_1_rho / safe_denom * v_val_interp
            
            return jnp.sum(integrand)
            
        return jax.vmap(compute_element)(jnp.arange(T), om_p_1d)
    
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    # Start with a random complex vector
    v = jax.random.normal(k1, (T,)) + 1j * jax.random.normal(k2, (T,))
    v = v / jnp.linalg.norm(v)
    
    for _ in range(n_iter):
        v = apply_Y(v)
        v_norm = jnp.linalg.norm(v)
        v = jnp.where(v_norm > 0, v / v_norm, v)
        
    # Scale to match roughly the data intensity
    power_scaling = jnp.sqrt(jnp.sum(jnp.abs(z_obs)))
    return v * power_scaling

def LBFGS_probe(
    om_p,
    om_x,
    om_t,
    z_obs,
    sigma_obs,
    rho_params,
    om_ref,
    n_probe_peaks=1, # unused, vector matches length instead
    obs_mask=None,
    maxiter=2000,
    tol=1e-6,
    rng_seed=0,
    lambda1=0.0,
    lambda2=0.0
):
    om_p = jnp.asarray(om_p)
    om_x = jnp.asarray(om_x)
    om_t = jnp.asarray(om_t)
    z_obs = jnp.asarray(z_obs)
    sigma_obs = jnp.asarray(sigma_obs)
    
    if obs_mask is None:
        mask = jnp.ones_like(z_obs, dtype=bool)
    else:
        mask = jnp.asarray(obs_mask, dtype=bool)
        
    om_grid = om_p[0, :] if om_p.ndim > 1 else om_p
    
    zero_ctx = _prepare_zero_omega_context(om_p, om_x, om_t, rho_params, om_ref)
    
    loss_fn = build_loss(om_grid, om_p, om_t, z_obs, sigma_obs, mask, zero_ctx, lambda1, lambda2)
    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=maxiter, tol=tol)
    
    v_init = spectral_initialization(om_grid, om_p, om_x, om_t, rho_params, om_ref, z_obs, sigma_obs, mask, n_iter=50)
    
    init_params = (jnp.real(v_init), jnp.imag(v_init))
    
    res = solver.run(init_params=init_params)

    jax.clear_caches()
    
    opt_f_real, opt_f_imag = res.params
    opt_f = opt_f_real + 1j * opt_f_imag
    
    z_fit = forward_model_discrete(opt_f_real, opt_f_imag, om_grid, om_p, om_t, zero_ctx)
    
    # Return discrete probe field over its spectral grid directly.
    return np.asarray(opt_f), np.asarray(z_fit)

