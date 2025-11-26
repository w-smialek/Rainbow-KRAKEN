using FFTW
using Plots
using DSP

# Reduced Planck constant in eV*fs (approx CODATA): ħ = 6.582119569e-16 eV·s
const ℏ = 6.582119569e-1

"""
    CFT(T_range, signal; use_window=true, inverse=false, zero_pad=0)

Continuous-time Fourier transform convention per column (fixed energy E):
    Forward: S(ω, E) = ∫ s(t, E) · e^{-i ω t} dt

Discrete implementation with FFT for samples t_n = t0 + n·dt:
    S(ω_k, E) ≈ dt · e^{-i ω_k t0} · FFT_n{ s(t_n, E) }[k]

Options:
  - use_window: apply Hann window (coherent-gain normalized) along time (rows).
  - inverse: when true, perform the inverse operation to reconstruct s(t, E)
             from S(ω, E). The input "signal" must be the shifted spectrum
             (as returned by the forward path). Windowing is not undone.
  - zero_pad: integer number of zeros to pad on BOTH sides along time before
              transform (forward) or to account for when inverting. Default 0.

Returns (forward, inverse respectively):
  - forward: spec_shift, OM_T, E_min, E_max
  - inverse: sig_time (trimmed to length(T_range) along rows), nothing, t_min, t_max
"""
function CFT(T_range, signal; use_window=true, inverse=false, zero_pad=0)
    # Validate input and infer sizes locally
    t = vec(T_range)
    length(t) >= 1 || error("T_range must be a 1D array of time samples")
    NT_local = length(t)
    NE_local = size(signal, 2)

    # Ensure uniform sampling
    NT_local >= 2 || error("T_range must contain at least two samples")
    dt = t[2] - t[1]
    all(isapprox.(diff(t), dt; rtol=1e-6, atol=0.0)) || error("T_range must be uniformly spaced")

    # Inverse path: reconstruct time-domain from shifted spectrum
    if inverse
        # Infer padded length from provided spectrum along rows
        S_shift = signal
        size(S_shift, 1) >= 1 || error("For inverse, 'signal' must be a 2D array with rows as ω bins")
        N_eff = size(S_shift, 1)

        # Frequency grid associated with the padded length
        freqs = fftfreq(N_eff, 1.0/dt)
        omega = 2π .* freqs

        # Undo shift and the forward phase factor (use padded start time)
        t0_eff = t[1] - Int(zero_pad) * dt
        S_unshift = ifftshift(S_shift, 1)
        phase_inv = exp.(1im .* omega .* t0_eff)
        S_unphased = S_unshift .* phase_inv

        # Inverse scaling: x ≈ (1/dt) · IFFT{ S }
        sig_time_full = ifft(S_unphased, 1) ./ dt

        # Trim padding back to original time length
        zp = Int(zero_pad)
        if zp > 0
            lo = zp + 1
            hi = zp + NT_local
            sig_time = sig_time_full[lo:hi, :]
        else
            sig_time = sig_time_full
        end

        # Return time-domain signal and simple time metadata
        return sig_time, nothing, Float64(t[1]), Float64(t[end])
    end

    # Forward path
    size(signal, 1) == NT_local || error("signal shape[1] must match length(T_range)")

    # Windowing (apply on original length), then symmetric zero padding
    if use_window
        window = hanning(NT_local)
        coherent_gain = mean(window)  # Hann coherent gain = 0.5
    else
        window = ones(NT_local)
        coherent_gain = 1.0
    end

    windowed = (signal .* window) ./ coherent_gain

    zp = Int(zero_pad)
    if zp > 0
        windowed = vcat(zeros(ComplexF64, zp, NE_local), windowed, zeros(ComplexF64, zp, NE_local))
    end
    N_eff = size(windowed, 1)

    # Frequencies (unshifted) and corresponding angular frequencies for padded length
    freqs = fftfreq(N_eff, 1.0/dt)   # cycles per unit time
    omega = 2π .* freqs              # rad / unit time

    # FFT along time axis with dt scaling to approximate the continuous integral
    spec = fft(windowed, 1) .* dt

    # Phase correction for start time including leading padding
    t0_eff = t[1] - zp * dt
    phase = exp.(-1im .* omega .* t0_eff)
    spec .*= phase

    # Shift spectrum and build energy axis (ħ·ω)
    spec_shift = fftshift(spec, 1)
    energy_axis = ℏ .* omega
    energy_axis_shift = fftshift(energy_axis)

    OM_T = energy_axis_shift ./ ℏ  # back to angular frequency ω
    OM_T = repeat(OM_T, 1, NE_local)

    return spec_shift, OM_T, energy_axis_shift[1], energy_axis_shift[end]
end

normalize_rho(rho) = rho ./ sum(abs.(diag(rho)))

normalize_abs(arr) = arr ./ sum(abs.(arr))

function spectrum_fun(A, om0, s, om)
    return A ./ (2 .* s) .* exp.(-(om .- om0).^2 ./ (2 .* s.^2))
end

function sp_tot(gausses, om)
    retval = zero(om)
    for gauss in gausses
        _, a0, om0, s0, _ = gauss
        retval .+= spectrum_fun(a0, om0, s0, om)
    end
    return retval
end

function synth_baseline(sp_pr, sp_x, om_pr, om_x, T)
    phase0 = exp.(1im .* 0 .* T)
    phase1 = exp.(1im .* om_pr .* T)
    phase2 = exp.(-1im .* om_x .* T)

    f1 = sp_pr .* phase0
    f2 = -sp_x ./ om_x .* phase2
    conv1 = conv_fft_2d(f1, f2)

    f1 = -sp_pr ./ om_pr .* phase1
    f2 = sp_x .* phase0
    conv2 = conv_fft_2d(f1, f2)

    return conv1 .+ conv2
end

function synth_baseline_hermit(input, sp_x, om_pr, om_x, T)
    phase0 = exp.(1im .* 0 .* T)
    phase1 = exp.(1im .* om_pr .* T)
    phase2 = exp.(-1im .* om_x .* T)

    f1 = input
    f2 = -sp_x ./ om_x .* conj.(phase2)
    conv1 = conv_fft_2d(f1, reverse(f2, dims=2))

    f1 = input
    f2 = sp_x .* phase0
    conv2 = conv_fft_2d(f1, reverse(f2, dims=2))
    conv2 = -conv2 ./ om_pr .* conj.(phase1)

    return vec(sum(conv1 .+ conv2, dims=1))
end

# Helper function for 2D FFT convolution along dimension 2 (columns)
function conv_fft_2d(a, b)
    # Convolve along dimension 2 (energy axis), same mode
    N = size(a, 2)
    result = similar(a)
    for i in 1:size(a, 1)
        result[i, :] = conv(vec(a[i, :]), vec(b[i, :]))[1:N]  # truncate to same
    end
    return result
end

###
### FIELD PARAMETERS
###

E_lo = 60.0
E_hi = 63.5
T_reach = 100.0
E_span = E_hi - E_lo

N_E = 900
N_T = 900

E_range = range(E_lo, E_hi, length=N_E)
T_range = range(-T_reach, T_reach, length=N_T)
E = repeat(collect(E_range)', N_T, 1)
T = repeat(collect(T_range), 1, N_E)

A_xuv = 1.0
om_xuv = 60.65 / ℏ
s_xuv = 0.15 / ℏ
pulse_xuv = (0 .* T, A_xuv, om_xuv, s_xuv, 0)

A_probe = 1.0
om_probe = 1.55 / ℏ
s_probe = 0.15 / ℏ
pulse_probe = (T, A_probe, om_probe, s_probe, 0)

A_probe2 = 0.2
om_probe2 = 1.20 / ℏ
s_probe2 = 0.04 / ℏ
pulse_probe2 = (T, A_probe2, om_probe2, s_probe2, 0)

A_probe3 = 0.2
om_probe3 = 2.0 / ℏ
s_probe3 = 0.07 / ℏ
pulse_probe3 = (T, A_probe3, om_probe3, s_probe3, 0)

A_probe4 = 0.3
om_probe4 = 1.85 / ℏ
s_probe4 = 0.17 / ℏ
pulse_probe4 = (T, A_probe4, om_probe4, s_probe4, 0)

A_ref = 1.0
om_ref = 1.55 / ℏ
s_ref = 0.005 / ℏ
pulse_ref = (0 .* T, A_ref, om_ref, s_ref, 0)

refs = (pulse_ref,)
probes = (pulse_probe, pulse_probe2, pulse_probe3, pulse_probe4)
xuvs = (pulse_xuv,)
refprobes = (refs..., probes...)

###
### CONTROL SAMPLE - DECONVOLUTION USING EXACT DATA
###

# Synthetic baseline with known spectra
om_probe_vals = vec((E ./ ℏ .- E_lo / ℏ .+ 0.1 .+ E_span / ℏ / N_E / 2 .* ((N_E - 1) % 2))[1, :])
sp_probe = sp_tot(probes, om_probe_vals)
sp_ref = sp_tot(refs, om_probe_vals)

om_xuv_vals = vec((E ./ ℏ .- E_span / ℏ / 2 .- 0.1)[1, :])
sp_xuv = sp_tot(xuvs, om_xuv_vals)

# synth_bsln = abs.(synth_baseline(sp_probe, sp_xuv, om_probe_vals, om_xuv_vals, T) .+ synth_baseline(sp_ref, sp_xuv, om_probe_vals, om_xuv_vals, T .* 0)).^2
synth_bsln = abs.(synth_baseline(sp_probe, sp_xuv, om_probe_vals, om_xuv_vals, T)).^2
synth_bsln_FT, _, _, _ = CFT(collect(T_range), synth_bsln, use_window=false)
# CHECKED - PRODUCES SAME AS 'AMPLITUDE'

w = ones(ComplexF64, length(sp_probe)) ./ sqrt(length(sp_probe))

n_power_iter = 10
m = length(synth_bsln)

for i_iter in 1:n_power_iter
    global w = (1 / m) .* synth_baseline_hermit(synth_bsln .* synth_baseline(w, sp_xuv, om_probe_vals, om_xuv_vals, T), sp_xuv, om_probe_vals, om_xuv_vals, T)
    w ./= sqrt(sum(abs.(w).^2))
end

lambda_bsln = synth_baseline(w, sp_xuv, om_probe_vals, om_xuv_vals, T)
alpha = sum(sqrt.(synth_bsln) .* abs.(lambda_bsln)) / sum(abs.(lambda_bsln).^2)

w = alpha .* w

p_initial = plot(real.(w), label="Re(w)")
plot!(p_initial, imag.(w), label="Im(w)")
plot!(p_initial, sp_probe, label="sp_probe")
title!(p_initial, "Initial spectrum estimate")
savefig(p_initial, "initial_spectrum_estimate.png")
println("Saved initial spectrum estimate to initial_spectrum_estimate.png")

n_main_iter = 1000

mu_step_max = 0.005
I_warmup = 200

normsq_z0 = sum(abs.(w).^2)

z = copy(w)

ers = Float64[]

for i_iter in 1:n_main_iter
    mu_step = mu_step_max * (1 - exp(-i_iter / I_warmup))

    push!(ers, sum(abs.(z .- sp_probe).^2) / sum(abs.(sp_probe).^2))

    sbforward = synth_baseline(z, sp_xuv, om_probe_vals, om_xuv_vals, T)
    sbhermit_arg = (abs.(sbforward).^2 .- synth_bsln) .* sbforward
    z .-= (mu_step / normsq_z0 / m) .* synth_baseline_hermit(sbhermit_arg, sp_xuv, om_probe_vals, om_xuv_vals, T)

    println(i_iter)

    if i_iter % 30 == 0
        p1 = plot(real.(z), label="Re(z)")
        plot!(p1, imag.(z), label="Im(z)")
        plot!(p1, sp_probe, label="sp_probe")
        title!(p1, "Current spectrum estimate")

        p2 = plot(log.(ers), label="Relative MSE")
        title!(p2, "log Error history")
        xlabel!(p2, "Iteration")

        combined_plot = plot(p1, p2, layout=(2, 1), size=(800, 600))
        filename = "spectrum_convergence_iter.png"
        savefig(combined_plot, filename)
        println("Saved convergence plot")
    end
end