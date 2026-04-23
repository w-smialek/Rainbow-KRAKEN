import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib import ticker
from matplotlib import font_manager as fm
from pathlib import Path
hbar = 6.582119569e-1


font_path1 = Path(__file__).resolve().parent / 'fonts' / 'cmu.sans-serif-bold.ttf'
font_path2 = Path(__file__).resolve().parent / 'fonts' / 'cmu.sans-serif-demi-condensed-demicondensed.ttf'
fm.fontManager.addfont(str(font_path1))
fm.fontManager.addfont(str(font_path2))
cmu_sans_bold = fm.FontProperties(fname=str(font_path1)).get_name()
cmu_sans = fm.FontProperties(fname=str(font_path2)).get_name()
plt.rcParams['font.family'] = cmu_sans_bold
# plt.rcParams['font.family'] = cmu_sans
# plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rc('axes', unicode_minus=False)

def _phase_eps(complex_mat):
	return complex_mat - 1e-4*np.mean(np.abs(complex_mat))

def plot_spectra(om_pr,
				 sp_pr,
				 sp_ref=None,
				 om_x=None,
				 sp_x=None,
				 save_path=None,
				 show=False,
				 title='IR spectrum and photelectron signal',
				 x_label='Energy [eV]',
				 y_label='Amplitude [arb. u.]',
				 phase_label='Phase [rad]',
				 probe_title='Probe and Reference Spectra',
				 xuv_title=r'Photoelectron populations $\rho(\varepsilon,\varepsilon)$',
				 probe_xlim=(0.8, 2.5),
				 xuv_xlim=(24.25, 25.75),
				 phase_ticks=None,
				 phase_tick_labels=None,
				 phase_threshold=0.05,
				 show_ref_phase=False,
				 if_square=False):

	has_second_axis = sp_x is not None
	if has_second_axis:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
	else:
		fig, ax1 = plt.subplots(figsize=(6.5, 5))
		ax2 = None

	probe_abs = np.abs(sp_pr)
	probe_phase = np.angle(sp_pr)

	line_probe_abs, = ax1.plot(om_pr * hbar, probe_abs, label='Probe |spectrum|', linewidth=2)

	line_ref_abs = None
	if sp_ref is not None:
		ref_abs = np.abs(sp_ref)
		line_ref_abs, = ax1.plot(om_pr * hbar, ref_abs, label='Reference |spectrum|', linewidth=2)

	ax1_phase = ax1.twinx()
	probe_phase_mask = probe_abs >= phase_threshold * np.max(probe_abs)
	probe_phase_plot = np.where(probe_phase_mask, probe_phase, np.nan)
	line_probe_phase, = ax1_phase.plot(
		om_pr * hbar,
		probe_phase_plot,
		'--',
		linewidth=1.2,
		alpha=0.8,
		label='Probe phase',
		color='tab:blue',
	)

	line_ref_phase = None
	if sp_ref is not None and show_ref_phase:
		ref_phase = np.angle(sp_ref)
		ref_phase_mask = ref_abs >= phase_threshold * np.max(ref_abs)
		ref_phase_plot = np.where(ref_phase_mask, ref_phase, np.nan)
		line_ref_phase, = ax1_phase.plot(
			om_pr * hbar,
			ref_phase_plot,
			'--',
			linewidth=1.2,
			alpha=0.8,
			label='Reference phase',
			color='tab:orange',
		)

	if phase_ticks is None:
		phase_ticks = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
	if phase_tick_labels is None:
		phase_tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$']

	ax1.set_xlabel(x_label)
	ax1.set_ylabel(y_label)
	ax1.set_title(probe_title, fontsize=12)
	ax1.set_xlim(probe_xlim)
	ax1.grid(True, alpha=0.3)

	ax1_phase.set_ylabel(phase_label)
	ax1_phase.set_ylim([-np.pi, np.pi])
	ax1_phase.set_yticks(phase_ticks)
	ax1_phase.set_yticklabels(phase_tick_labels)

	legend_lines = [line_probe_abs, line_probe_phase]
	if line_ref_abs is not None:
		legend_lines.append(line_ref_abs)
	if line_ref_phase is not None:
		legend_lines.append(line_ref_phase)
	legend_labels = [line.get_label() for line in legend_lines]
	ax1.legend(legend_lines, legend_labels, loc='best')

	if has_second_axis:
		ax2.plot(om_x * hbar, sp_x, label='Photoelectron populations', linewidth=2, color='purple')
		ax2.set_xlabel(x_label)
		ax2.set_ylabel('Signal strength [arb. u.]')
		ax2.set_title(xuv_title, fontsize=12)
		ax2.set_xlim(xuv_xlim)
		ax2.grid(True, alpha=0.3)

	if if_square:
		ax1.set_box_aspect(1)
		if ax2 is not None:
			ax2.set_box_aspect(1)

	fig.suptitle(title, fontsize=20, weight='bold')
	fig.tight_layout()
	if save_path is not None:
		fig.savefig(save_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

	return

def abs_plot(
	mat_abs,
	extent,
	save_path=None,
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	y_ticks=None,
	y_tick_labels=None,
	if_square=False,
	magnitude_cmap='viridis',
):
	# Compute magnitude for plotting
	magnitude = np.abs(mat_abs)

	fig, ax = plt.subplots(figsize=(6.5, 5))
	fig.suptitle(title, fontsize=18, font=cmu_sans_bold)

	# Plot Magnitude
	im1 = ax.imshow(magnitude, origin='lower', extent=extent, aspect='auto', cmap=magnitude_cmap)
	ax.set_title('Magnitude', fontsize=14)
	ax.set_xlabel(x_label, fontsize=12)
	ax.set_ylabel(y_label, fontsize=12)
	if y_ticks is not None:
		ax.set_yticks(y_ticks)
	if y_tick_labels is not None:
		ax.set_yticklabels(y_tick_labels)
	if if_square:
		ax.set_box_aspect(1)
	ax.grid(True, which='major', linestyle='--', linewidth=0.25, color='white', alpha=0.35)
	cbar1 = plt.colorbar(im1, ax=ax, fraction=0.2, pad=0.0)
	cbar1.ax.set_title('Intensity', fontsize=8, pad=4)

	sci_formatter = ticker.ScalarFormatter(useMathText=True)
	sci_formatter.set_scientific(True)
	sci_formatter.set_powerlimits((0, 0))
	cbar1.formatter = sci_formatter
	cbar1.update_ticks()
	cbar1.ax.yaxis.get_offset_text().set_size(8)
	# cbar1.ax.yaxis.get_offset_text().set_fontfamily('DejaVu Sans')

	fig.tight_layout()
	if save_path is not None:
		fig.savefig(save_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

	return

def complex_plot(
	mat_complex,
	extent,
	save_path=None,
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	y_ticks=None,
	y_tick_labels=None,
	if_square=False,
	magnitude_cmap='viridis',
	phase_cmap='twilight_shifted',
):
	# Compute magnitude and phase for plotting
	magnitude = np.abs(mat_complex)
	phase = np.angle(_phase_eps(mat_complex))

	fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 5),sharey=True,gridspec_kw={'width_ratios': [1, 1]})
	fig.suptitle(title, fontsize=18, fontweight='bold',font=cmu_sans_bold)

	# Plot Magnitude
	im1 = ax1.imshow(magnitude, origin='lower', extent=extent, aspect='auto', cmap=magnitude_cmap)
	ax1.set_title('Magnitude', fontsize=14)
	ax1.set_xlabel(x_label, fontsize=12)
	ax1.set_ylabel(y_label, fontsize=12)
	if y_ticks is not None:
		ax1.set_yticks(y_ticks)
	if y_tick_labels is not None:
		ax1.set_yticklabels(y_tick_labels)
	if if_square:
		ax1.set_box_aspect(1)
		ax2.set_box_aspect(1)
	ax1.grid(True, which='major', linestyle='--', linewidth=0.25, color='white', alpha=0.35)
	cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.2, pad=0.0)
	cbar1.ax.set_title('Intensity', fontsize=8, pad=4)

	sci_formatter = ticker.ScalarFormatter(useMathText=True)
	sci_formatter.set_scientific(True)
	sci_formatter.set_powerlimits((0, 0))
	cbar1.formatter = sci_formatter
	cbar1.update_ticks()
	cbar1.ax.yaxis.get_offset_text().set_size(8)

	# Plot Phase
	im2 = ax2.imshow(phase, origin='lower', extent=extent, aspect='auto', cmap=phase_cmap)
	ax2.set_title('Phase', fontsize=14)
	ax2.set_xlabel(x_label, fontsize=12)
	ax2.grid(True, which='major', linestyle='--', linewidth=0.25, color='white', alpha=0.35)
	ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
	cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.2, pad=0.0)
	cbar2.ax.set_title('Phase\n(rad)', fontsize=8, pad=4)
	phase_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
	phase_tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$']
	cbar2.set_ticks(phase_ticks)
	cbar2.set_ticklabels(phase_tick_labels)

	fig.tight_layout()
	if save_path is not None:
		fig.savefig(save_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

	return

###
## 1GENERATE_SIGNAL
###

file_path = 'single_output_temp/1generate_signal/input_spectra.npz'
input_spectra = np.load(file_path)

plot_spectra(
			om_pr = input_spectra['om_probe'],
			sp_pr = input_spectra['sp_probe'],
			sp_ref = input_spectra['sp_ref'],
			om_x = input_spectra['om_xuv'],
			sp_x = input_spectra['sp_xuv'],
			save_path = 'plot_output/1generate_signal/input_spectra.png',
			show=False,
)

file_path = 'single_output_temp/1generate_signal/exact_freqsig.npz'
exact_freqsig = np.load(file_path)

complex_plot(
	mat_complex=exact_freqsig['mat_complex'],
	extent=exact_freqsig['extent'],
	save_path='plot_output/1generate_signal/exact_freqsig.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	y_ticks=[-3.1,-1.55,0,1.55,3.1],
	y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

file_path = 'single_output_temp/1generate_signal/exact_timesig.npz'
exact_timesig = np.load(file_path)

abs_plot(
	mat_abs=exact_timesig['mat_abs'],
	extent=exact_timesig['extent'],
	save_path='plot_output/1generate_signal/exact_timesig.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo'
)

file_path = 'single_output_temp/1generate_signal/measured_timesig.npz'
measured_timesig = np.load(file_path)

abs_plot(
	mat_abs=measured_timesig['mat_abs'],
	extent=measured_timesig['extent'],
	save_path='plot_output/1generate_signal/measured_timesig.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo'
)

###
## 2PROCESS_DETREND
###

file_path = 'single_output_temp/2process_detrend/measured_freqsig.npz'
measured_freqsig = np.load(file_path)

complex_plot(
	mat_complex=measured_freqsig['mat_complex'],
	extent=measured_freqsig['extent'],
	save_path='plot_output/2process_detrend/measured_freqsig.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	y_ticks=[-3.1,-1.55,0,1.55,3.1],
	y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

###
## 3KB_CORRECT
###

file_path = 'single_output_temp/3kb_correct/KB_before.npz'
KB_before = np.load(file_path)

complex_plot(
	mat_complex=KB_before['mat_complex'],
	extent=KB_before['extent'],
	save_path='plot_output/3kb_correct/KB_before.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

file_path = 'single_output_temp/3kb_correct/KB_after.npz'
KB_after = np.load(file_path)

complex_plot(
	mat_complex=KB_after['mat_complex'],
	extent=KB_after['extent'],
	save_path='plot_output/3kb_correct/KB_after.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

file_path = 'single_output_temp/3kb_correct/zero_omega_comp.npz'
zero_omega_comp = np.load(file_path)

complex_plot(
	mat_complex=zero_omega_comp['mat_complex'],
	extent=zero_omega_comp['extent'],
	save_path='plot_output/3kb_correct/zero_omega_comp.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

file_path = 'single_output_temp/3kb_correct/zero_omega_comp_sigma.npz'
zero_omega_comp_sigma = np.load(file_path)

abs_plot(
	mat_abs=zero_omega_comp_sigma['mat_abs'],
	extent=zero_omega_comp_sigma['extent'],
	save_path='plot_output/3kb_correct/zero_omega_comp_sigma.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo'
)

###
## 4PROBE_REC
###

file_path = 'single_output_temp/4probe_rec/probe_sp_rec.npz'
probe_sp_rec = np.load(file_path)

plot_spectra(probe_sp_rec['om_probe'],
			 probe_sp_rec['sp_probe'],
			 sp_ref=probe_sp_rec['sp_probe_rec'],
			 save_path='plot_output/4probe_rec/probe_sp_rec.png',
			 title='IR spectrum and photelectron signal',
			 x_label='Energy [eV]',
			 y_label='Amplitude [arb. u.]',
			 phase_label='Phase [rad]',
			 probe_title='Probe and Reference Spectra',
			 probe_xlim=(1.0, 2.0),
			 phase_ticks=None,
			 phase_tick_labels=None,
			 phase_threshold=0.05,
			 show_ref_phase=True)

###
## 5PROBE_CORR
###

file_path = 'single_output_temp/5probe_corr/data_rho.npz'
data_rho = np.load(file_path)

complex_plot(
	mat_complex=data_rho['mat_complex'],
	extent=data_rho['extent'],
	save_path='plot_output/5probe_corr/data_rho.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

file_path = 'single_output_temp/5probe_corr/data_rho_rec.npz'
data_rho_rec = np.load(file_path)

complex_plot(
	mat_complex=data_rho_rec['mat_complex'],
	extent=data_rho_rec['extent'],
	save_path='plot_output/5probe_corr/data_rho_rec.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

file_path = 'single_output_temp/5probe_corr/data_sigma.npz'
data_sigma = np.load(file_path)

abs_plot(
	mat_abs=data_sigma['mat_abs'],
	extent=data_sigma['extent'],
	save_path='plot_output/5probe_corr/data_sigma.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo'
)

file_path = 'single_output_temp/5probe_corr/data_sigma_rec.npz'
data_sigma_rec = np.load(file_path)

abs_plot(
	mat_abs=data_sigma_rec['mat_abs'],
	extent=data_sigma_rec['extent'],
	save_path='plot_output/5probe_corr/data_sigma_rec.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo'
)

###
## 6MCMC
###

file_path = 'single_output_temp/6mcmc/rho_ideal.npz'
rho_ideal = np.load(file_path)

complex_plot(
	mat_complex=rho_ideal['mat_complex'],
	extent=rho_ideal['extent'],
	save_path='plot_output/6mcmc/rho_ideal.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

#
# INTERP
#

file_path = 'single_output_temp/6mcmc/data_rho_interp.npz'
data_rho_interp = np.load(file_path)

complex_plot(
	mat_complex=data_rho_interp['mat_complex'],
	extent=data_rho_interp['extent'],
	save_path='plot_output/6mcmc/data_rho_interp.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

file_path = 'single_output_temp/6mcmc/data_rho_interp_rec.npz'
data_rho_interp_rec = np.load(file_path)

complex_plot(
	mat_complex=data_rho_interp_rec['mat_complex'],
	extent=data_rho_interp_rec['extent'],
	save_path='plot_output/6mcmc/data_rho_interp_rec.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

file_path = 'single_output_temp/6mcmc/data_sigma_interp.npz'
data_sigma_interp = np.load(file_path)

abs_plot(
	mat_abs=data_sigma_interp['mat_abs'],
	extent=data_sigma_interp['extent'],
	save_path='plot_output/6mcmc/data_sigma_interp.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo'
)

file_path = 'single_output_temp/6mcmc/data_sigma_interp_rec.npz'
data_sigma_interp_rec = np.load(file_path)

abs_plot(
	mat_abs=data_sigma_interp_rec['mat_abs'],
	extent=data_sigma_interp_rec['extent'],
	save_path='plot_output/6mcmc/data_sigma_interp_rec.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo'
)

#
# INFERRED
#

file_path = 'single_output_temp/6mcmc/rho_inferred.npz'
rho_inferred = np.load(file_path)

complex_plot(
	mat_complex=rho_inferred['mat_complex'],
	extent=rho_inferred['extent'],
	save_path='plot_output/6mcmc/rho_inferred.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)

file_path = 'single_output_temp/6mcmc/rho_inferred_rec.npz'
rho_inferred_rec = np.load(file_path)

complex_plot(
	mat_complex=rho_inferred_rec['mat_complex'],
	extent=rho_inferred_rec['extent'],
	save_path='plot_output/6mcmc/rho_inferred_rec.png',
	show=False,
	title='Complex Array Plot',
	x_label='X-axis Label (Placeholder)',
	y_label='Y-axis Label (Placeholder)',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)