import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib import ticker
from matplotlib import font_manager as fm
from pathlib import Path
import re
hbar = 6.582119569e-1


font_path1 = './fonts/cmu.sans-serif-bold.ttf'
font_path2 = './fonts/cmu.sans-serif-demi-condensed-demicondensed.ttf'
fm.fontManager.addfont(str(font_path1))
fm.fontManager.addfont(str(font_path2))
cmu_sans_bold = fm.FontProperties(fname=str(font_path1)).get_name()
cmu_sans = fm.FontProperties(fname=str(font_path2)).get_name()
plt.rcParams['font.family'] = [cmu_sans_bold, 'DejaVu Sans']
# plt.rcParams['font.family'] = cmu_sans
# plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rc('axes', unicode_minus=False)

def _phase_eps(complex_mat):
	return complex_mat - 1e-4*np.mean(np.abs(complex_mat))

def _circular_mean(angles, axis=0):
    """Return circular mean for angles in radians, robust to +/-pi wrap."""
    angles = np.asarray(angles)
    if angles.size == 0:
        return np.array([])
    return np.angle(np.mean(np.exp(1j * angles), axis=axis))


def _replace_offset_text_with_manual_copy(cbar_axis):
	"""Capture the finalized offset text, hide the managed artist, and redraw it manually."""
	offset_text = cbar_axis.yaxis.get_offset_text()
	offset_string = offset_text.get_text()
	if not offset_string:
		return

	fig = cbar_axis.figure
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	bbox = offset_text.get_window_extent(renderer=renderer)

	# Capture the finalized layout state before detaching the artist.
	fontsize = offset_text.get_size()
	color = offset_text.get_color()
	rotation = offset_text.get_rotation()
	fontfamily = offset_text.get_fontfamily()

	# # Move the exponent outside the mathdefault-wrapped 10 so the exponent uses the native math font.
	offset_string = re.sub(
			r'\\mathdefault\{10\^\{(-?)(\d+)\}\}',
			r'\\mathdefault{10}^{\1\\mathdefault{\2}}',
			offset_string)

	# Hide the original offset text so savefig/draw does not render it again.
	# offset_text.set_visible(False)
	offset_text.set_alpha(0.01)


	manual_x, manual_y = fig.transFigure.inverted().transform((bbox.x0, bbox.y0))
	manual_text = fig.text(
		manual_x,
		manual_y,
		offset_string,
		fontsize=fontsize,
		color=color,
		rotation=rotation,
		ha='left',
		va='bottom',
		clip_on=False,
	)
	manual_text.set_fontfamily(fontfamily)

def _rewrite_posterior_keys(npz_data):
	"""Rewrite saved MCMC posterior keys to readable physics-style names."""
	renamed = {}
	for key in npz_data.files:
		values = npz_data[key]

		match = re.match(r'^amps_(\d+)$', key)
		if match:
			k = int(match.group(1)) + 1
			renamed[f'$C_{{{k}{k}}}$'] = values
			continue

		match = re.match(r'^mus_(\d+)$', key)
		if match:
			k = int(match.group(1)) + 1
			renamed[f'$\\mu_{k}$'] = values
			continue

		match = re.match(r'^sigmas_(\d+)$', key)
		if match:
			k = int(match.group(1)) + 1
			renamed[f'$\\sigma_{k}$'] = values
			continue

		match = re.match(r'^betas_(\d+)$', key)
		if match:
			k = int(match.group(1)) + 1
			renamed[f'$\\beta_{k}$'] = values
			continue

		match = re.match(r'^taus_(\d+)$', key)
		if match:
			k = int(match.group(1)) + 1
			renamed[f'$\\tau_{k}$'] = values
			continue

		match = re.match(r'^lambdas_(\d+)$', key)
		if match:
			k = int(match.group(1)) + 1
			renamed[f'$\\gamma_{k}$'] = values
			continue

		match = re.match(r'^gamma_mag_(\d+)_(\d+)$', key)
		if match:
			i = int(match.group(1)) + 1
			j = int(match.group(2)) + 1
			renamed[f'$|C_{{{i}{j}}}| / \\sqrt{{C_{{{i}{i}}} C_{{{j}{j}}}}}$'] = values
			continue

		match = re.match(r'^gamma_phase_(\d+)_(\d+)$', key)
		if match:
			i = int(match.group(1)) + 1
			j = int(match.group(2)) + 1
			renamed[f'$\\text{{arg}}(C_{{{i}{j}}})$'] = values
			continue

		match = re.match(r'^eta_offdiag_(\d+)_(\d+)$', key)
		if match:
			i = int(match.group(1)) + 1
			j = int(match.group(2)) + 1
			renamed[f'$\\eta_{{{i}{j}}}$'] = values
			continue

		# Keep unknown keys untouched.
		renamed[key] = values

	return renamed

def plot_posterior(posterior_panels,save_path=None):
	if isinstance(posterior_panels, dict):
		posterior_panels = list(posterior_panels.items())

	color_hist = "#231696" # 'tab:blue'

	n_panels = len(posterior_panels)
	n_cols = 4
	n_rows = int(np.ceil(n_panels / n_cols))
	fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.8 * n_rows))
	axs = np.atleast_1d(axs).ravel()

	for ax, (label, values) in zip(axs, posterior_panels):
		if 'arg' in label:
			values_wrapped = ((values + np.pi) % (2.0 * np.pi)) - np.pi
			ax.hist(values_wrapped, bins=np.linspace(-np.pi, np.pi, 41), density=False, alpha=0.75, color=color_hist)
			ax.axvline(_circular_mean(values_wrapped), color='black', linestyle='--', linewidth=1.0)
			ax.set_xlim(-np.pi, np.pi)
		else:
			ax.hist(values, bins=40, density=False, alpha=0.75, color=color_hist)
			ax.axvline(np.mean(values), color='black', linestyle='--', linewidth=1.0)

		# ax.grid(True, alpha=0.3)
		ax.grid(True, which='major', linestyle='--', linewidth=0.4, color='gray', alpha=0.35)

		x_min, x_max = ax.get_xlim()
		if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
			# Keep labels readable for narrow ranges: fewer ticks, max 3 decimals.
			ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=4))
			ticks = ax.get_xticks()
			ticks = ticks[(ticks >= x_min) & (ticks <= x_max)]
			decimals = 0
			for d in range(4):
				if np.allclose(ticks * (10 ** d), np.round(ticks * (10 ** d)), atol=1e-8):
					decimals = d
					break
			else:
				decimals = 3
			ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(f'%.{decimals}f'))
		ax.set_title(label, fontsize=20, y=1.04)

	for ax in axs[n_panels:]:
		ax.axis('off')

	fig.suptitle('Posterior Distributions of Fitted Parameters', fontsize=30, y=1.02, weight='bold')
	plt.tight_layout()
	if save_path is not None:
		plt.savefig(save_path, dpi=200, bbox_inches='tight')
	plt.close()

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

	# IR/Pump branch (infrared domain) - warm colors
	color_probe = '#C41E3A' # '#D62728'      # Strong red-orange (crimson) - PRIMARY IR signal
	color_reference = '#0047AB' # '#1F77B4'  # Professional blue - REFERENCE/CALIBRATION

	# XUV/Probe branch (extreme UV domain) - distinct cool color  
	color_xuv = '#6A0572' # '#2CA02C'        # Mossy green - clearly distinct from IR, represents higher energy

	line_probe_abs, = ax1.plot(om_pr * hbar, probe_abs, 
							label='Probe |spectrum|', linewidth=2, color=color_probe)

	line_ref_abs = None
	if sp_ref is not None:
		ref_abs = np.abs(sp_ref)
		line_ref_abs, = ax1.plot(om_pr * hbar, ref_abs, 
						   label='Reference |spectrum|', linewidth=2, color=color_reference)

	ax1_phase = ax1.twinx()
	probe_phase_mask = probe_abs >= phase_threshold * np.max(probe_abs)
	probe_phase_plot = np.where(probe_phase_mask, probe_phase, np.nan)
	line_probe_phase, = ax1_phase.plot(
		om_pr * hbar,
		probe_phase_plot,
		linestyle=(0, (1, 1)),
		linewidth=1.2,
		alpha=0.8,
		label='Probe phase',
		color=color_probe,
	)

	line_ref_phase = None
	if sp_ref is not None and show_ref_phase:
		ref_phase = np.angle(sp_ref)
		ref_phase_mask = ref_abs >= phase_threshold * np.max(ref_abs)
		ref_phase_plot = np.where(ref_phase_mask, ref_phase, np.nan)
		line_ref_phase, = ax1_phase.plot(
			om_pr * hbar,
			ref_phase_plot,
			linestyle=(0, (1, 1)),
			linewidth=1.2,
			alpha=0.8,
			label='Reference phase',
			color=color_reference,
		)

	if phase_ticks is None:
		phase_ticks = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
	if phase_tick_labels is None:
		phase_tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$']

	ax1.set_xlabel(x_label)
	ax1.set_ylabel(y_label)
	ax1.set_title(probe_title, fontsize=12)
	ax1.set_xlim(probe_xlim)
	# ax1.grid(True, alpha=0.3)
	ax1.grid(True, which='major', linestyle='--', linewidth=0.4, color='gray', alpha=0.35)

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
		ax2.plot(om_x * hbar, sp_x, label='Photoelectron populations', linewidth=2, color=color_xuv)
		ax2.set_xlabel(x_label)
		ax2.set_ylabel('Signal strength [arb. u.]')
		ax2.set_title(xuv_title, fontsize=12)
		ax2.set_xlim(xuv_xlim)
		# ax2.grid(True, alpha=0.3)
		ax2.grid(True, which='major', linestyle='--', linewidth=0.4, color='gray', alpha=0.35)

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

def plot_fids(s_ref_list,
				fids,
				save_path=None,
				show=False,
				title='IR spectrum and photelectron signal',
				x_label='Energy [eV]',
				y_label='Amplitude [arb. u.]',
				# probe_xlim=(0.8, 2.5),
				if_square=False):


	fig, ax1 = plt.subplots(figsize=(6.5, 5))
	ax2 = None

	# IR/Pump branch (infrared domain) - warm colors
	color_probe = '#C41E3A' # '#D62728'      # Strong red-orange (crimson) - PRIMARY IR signal

	line_probe_abs, = ax1.plot(s_ref_list * hbar, fids, 
							label='Probe |spectrum|', linewidth=2, color=color_probe)

	ax1.set_xlabel(x_label)
	ax1.set_ylabel(y_label)
	ax1.set_title(title, fontsize=12,y=1.03)
	# ax1.set_xlim(probe_xlim)
	ax1.grid(True, which='major', linestyle='--', linewidth=0.4, color='gray', alpha=0.35)

	if if_square:
		ax1.set_box_aspect(1)

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
	caption=None,
	captionsize=16,
	cbar_title='Magnitude'
):
	# Compute magnitude for plotting
	magnitude = np.abs(mat_abs)

	fig, ax = plt.subplots(figsize=(6.5, 5))
	# fig.suptitle(title, fontsize=18, font=cmu_sans_bold)

	# Plot Magnitude
	im1 = ax.imshow(magnitude, origin='lower', extent=extent, aspect='auto', cmap=magnitude_cmap)
	ax.set_title(title, fontsize=18,y=1.02)
	ax.set_xlabel(x_label, fontsize=12)
	ax.set_ylabel(y_label, fontsize=12)
	if y_ticks is not None:
		ax.set_yticks(y_ticks)
	if y_tick_labels is not None:
		ax.set_yticklabels(y_tick_labels)
	if if_square:
		ax.set_box_aspect(1)
	if caption is not None:
		ax.text(
			0.02,
			0.98,
			caption,
			transform=ax.transAxes,
			fontsize=captionsize,
			verticalalignment="top",
			horizontalalignment="left",
			color="white",
			weight="bold",
			path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
		)
	ax.grid(True, which='major', linestyle='--', linewidth=0.25, color='gray', alpha=0.35)
	cbar1 = plt.colorbar(im1, ax=ax, fraction=0.2, pad=0.0)
	cbar1.ax.set_title(cbar_title, fontsize=8, pad=4)

	sci_formatter = ticker.ScalarFormatter(useMathText=True)
	sci_formatter.set_scientific(True)
	sci_formatter.set_powerlimits((0, 0))
	cbar1.formatter = sci_formatter
	cbar1.update_ticks()

	fig.tight_layout()
	_replace_offset_text_with_manual_copy(cbar1.ax)

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
	caption=None,
	captionsize=16,
	cbar_title = 'Magnitude'
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
	ax1.grid(True, which='major', linestyle='--', linewidth=0.35, color='gray', alpha=0.35)
	cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.2, pad=0.0)
	cbar1.ax.set_title(cbar_title, fontsize=8, pad=4)

	sci_formatter = ticker.ScalarFormatter(useMathText=True)
	sci_formatter.set_scientific(True)
	sci_formatter.set_powerlimits((0, 0))
	cbar1.formatter = sci_formatter
	cbar1.update_ticks()

	if caption is not None:
		ax1.text(
			0.02,
			0.98,
			caption,
			transform=ax1.transAxes,
			fontsize=captionsize,
			verticalalignment="top",
			horizontalalignment="left",
			color="white",
			weight="bold",
			path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
		)

	# Plot Phase
	im2 = ax2.imshow(phase, origin='lower', extent=extent, aspect='auto', cmap=phase_cmap,vmin=-np.pi,vmax=np.pi)
	ax2.set_title('Phase', fontsize=14)
	ax2.set_xlabel(x_label, fontsize=12)
	ax2.grid(True, which='major', linestyle='--', linewidth=0.25, color='gray', alpha=0.35)
	ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
	cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.2, pad=0.0)
	cbar2.ax.set_title('Phase\n[rad]', fontsize=8, pad=4)
	phase_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
	phase_tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$']
	cbar2.set_ticks(phase_ticks)
	cbar2.set_ticklabels(phase_tick_labels)

	fig.tight_layout()
	_replace_offset_text_with_manual_copy(cbar1.ax)
	if save_path is not None:
		fig.savefig(save_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()
	return


def create_stacked_overview_figure(im_paths, heights, gaps, fig_aspect, dpi=300, save_path=None):
	"""Stack pre-rendered images row-by-row with explicit row heights and vertical gaps.

	Parameters
	----------
	im_paths : list[tuple[str, ...]]
		Each tuple defines one row and contains the image paths in that row.
	heights : list[float]
		Row heights (arbitrary units, one per row).
	gaps : list[float]
		Vertical gaps including outer margins (len(im_paths) + 1).
		gaps[0] is the top margin, gaps[-1] is the bottom margin,
		and interior values are row-to-row separations.
	fig_aspect : float
		Figure aspect ratio defined as width / height.
	dpi : int
		DPI used when saving.
	save_path : str | None
		Output path.
	"""

	if len(im_paths) != len(heights):
		raise ValueError('im_paths and heights must have the same length.')
	if len(gaps) != len(im_paths) + 1:
		raise ValueError('gaps must have length len(im_paths) + 1.')
	if fig_aspect <= 0:
		raise ValueError('fig_aspect must be positive.')

	rows = []
	for row_paths in im_paths:
		if len(row_paths) == 0:
			raise ValueError('Each row in im_paths must contain at least one image path.')
		row_imgs = [plt.imread(path) for path in row_paths]
		row_aspects = []
		for im in row_imgs:
			if im.ndim < 2:
				raise ValueError('Loaded image has invalid shape.')
			h, w = im.shape[:2]
			if h == 0:
				raise ValueError('Loaded image height is zero.')
			row_aspects.append(w / h)
		rows.append((row_imgs, row_aspects))

	total_height = float(np.sum(heights) + np.sum(gaps))
	if total_height <= 0:
		raise ValueError('Sum of heights and gaps must be positive.')

	figure_width_units = fig_aspect * total_height

	# Global uniform spacing used for all multi-image rows.
	hgap_limits = []
	for row_idx, (_, aspects) in enumerate(rows):
		n_images = len(aspects)
		if n_images <= 1:
			continue
		base_row_width = heights[row_idx] * float(np.sum(aspects))
		hgap_limits.append((figure_width_units - base_row_width) / (n_images - 1))

	if hgap_limits:
		hgap_max = min(hgap_limits)
		if hgap_max < 0:
			raise ValueError('fig_aspect is too small for the provided row heights and image aspects.')
		hgap = 0.9 * hgap_max
	else:
		hgap = 0.0

	fig_height_in = total_height
	fig_width_in = fig_aspect * fig_height_in
	fig = plt.figure(figsize=(fig_width_in, fig_height_in))
	fig.patch.set_alpha(0.0)

	y_top = total_height - gaps[0]
	for row_idx, (row_data, row_height, row_gap) in enumerate(zip(rows, heights, gaps[1:])):
		row_imgs, row_aspects = row_data
		n_images = len(row_imgs)
		row_width = row_height * float(np.sum(row_aspects)) + hgap * max(0, n_images - 1)
		if row_width > figure_width_units + 1e-12:
			raise ValueError('A row does not fit within figure width. Increase fig_aspect or reduce heights.')

		x_cursor = 0.5 * (figure_width_units - row_width)
		y_bottom = y_top - row_height

		for im, im_aspect in zip(row_imgs, row_aspects):
			ax_width = row_height * im_aspect
			box = [
				x_cursor / figure_width_units,
				y_bottom / total_height,
				ax_width / figure_width_units,
				row_height / total_height,
			]
			ax = fig.add_axes(box)
			ax.imshow(im)
			ax.set_axis_off()
			ax.set_facecolor((0, 0, 0, 0))
			x_cursor += ax_width + (hgap if n_images > 1 else 0.0)

		y_top = y_bottom - row_gap

	if save_path is not None:
		fig.savefig(save_path, dpi=dpi, transparent=True)
	plt.close()
	return

file_path = 'single_output_temp/6mcmc/rho_ideal.npz'
rho_ideal = np.load(file_path)

rho_ideal = complex_plot(
	mat_complex=rho_ideal['mat_complex'],
	extent=rho_ideal['extent'],
	save_path='parameter_scan/pscan_output/rho_ideal.png',
	show=False,
	title='Initial Photoelectron Density Matrix',
	x_label=r'Energy $\varepsilon_2$ [eV]',
	y_label=r'Energy $\varepsilon_1$ [eV]',
	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	magnitude_cmap='turbo',
	phase_cmap='twilight_shifted'
)


file_path = 'single_output_temp/1generate_signal/input_spectra.npz'
input_spectra = np.load(file_path)

input_spectra = plot_spectra(
			om_pr = input_spectra['om_probe'],
			sp_pr = input_spectra['sp_probe'],
			sp_ref = input_spectra['sp_ref'],
			om_x = input_spectra['om_xuv'],
			sp_x = input_spectra['sp_xuv'],
			save_path = 'parameter_scan/pscan_output/input_spectra.png',
			show=False,
			title='IR Spectrum and Photelectron Populations',
			x_label='Energy [eV]',
			y_label='Amplitude [arb. u.]',
			phase_label='Phase [rad]',
			probe_title='Probe and Reference Spectra',
			xuv_title=r'Photoelectron Populations $\rho(\varepsilon,\varepsilon)$',
)

for suffix in ['_it0','_it1']:

	file_path = f'single_output_temp/4probe_rec/probe_sp_rec{suffix}.npz'
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
	
	file_path = f'single_output_temp/6mcmc/rho_inferred{suffix}.npz'
	rho_inferred = np.load(file_path)

	complex_plot(
		mat_complex=rho_inferred['mat_complex'],
		extent=rho_inferred['extent'],
		save_path=f'parameter_scan/pscan_output/rho_inferred{suffix}.png',
		show=False,
		title='Inferred Density Matrix',
		x_label=r'Energy $\varepsilon_2$ [eV]',
		y_label=r'Energy $\varepsilon_1$ [eV]',
		# y_ticks=[-3.1,-1.55,0,1.55,3.1],
		# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
		magnitude_cmap='turbo',
		phase_cmap='twilight_shifted',
		caption = f'F = {rho_inferred['RES']:.3f}'
	)
	

exit()

s_ref_list = np.linspace(0.010,0.050,9) / hbar
s_ref_list = np.linspace(0.060,0.080,3) / hbar

# file_path = 'parameter_scan/s_ref_fids.npy'
# fids = np.load(file_path)

s_ref_list = np.array([0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.06,0.07,0.08])/hbar
fids = np.array([0.927,0.951,0.977,0.976,0.974,0.977,0.978,0.969,0.976,0.960,0.955,0.942])

input_spectra = plot_fids(
			s_ref_list,
			fids,
			save_path = f'parameter_scan/fids.png',
			show=False,
			title='Fids',
			x_label=r'$\hbar\sigma_r$ [eV]',
			y_label='Fidelity',
)
exit()

for s_ref in s_ref_list:

    suffix = f'_s_ref_{s_ref*hbar:.3f}'

    file_path = f'single_output_temp/1generate_signal/input_spectra{suffix}.npz'
    input_spectra = np.load(file_path)

    input_spectra = plot_spectra(
				om_pr = input_spectra['om_probe'],
				sp_pr = input_spectra['sp_probe'],
				sp_ref = input_spectra['sp_ref'],
				om_x = input_spectra['om_xuv'],
				sp_x = input_spectra['sp_xuv'],
				save_path = f'parameter_scan/pscan_output/input_spectra{suffix}.png',
				show=False,
				title='IR Spectrum and Photelectron Populations',
				x_label='Energy [eV]',
				y_label='Amplitude [arb. u.]',
				phase_label='Phase [rad]',
				probe_title='Probe and Reference Spectra',
				xuv_title=r'Photoelectron Populations $\rho(\varepsilon,\varepsilon)$',
	)

    file_path = f'single_output_temp/6mcmc/data_rho_interp{suffix}.npz'
    data_rho_interp = np.load(file_path)

    complex_plot(
		mat_complex=data_rho_interp['mat_complex'],
		extent=data_rho_interp['extent'],
		save_path=f'parameter_scan/pscan_output/data_rho_interp{suffix}.png',
		show=False,
		title=r'$\tilde S_{\text{corr}} (\varepsilon_2,\varepsilon_1)$',
		x_label=r'Energy $\varepsilon_2$ [eV]',
		y_label=r'Energy $\varepsilon_1$ [eV]',
		# y_ticks=[-3.1,-1.55,0,1.55,3.1],
		# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
		magnitude_cmap='turbo',
		phase_cmap='twilight_shifted'
	)

	# abs_plot(
	# 	mat_abs=np.abs(data_rho_interp['mat_complex']),
	# 	extent=data_rho_interp['extent'],
	# 	save_path='plot_output/6mcmc/data_rho_interp_abs.png',
	# 	show=False,
	# 	title=r'$ \left| \tilde S_{\text{corr}} (\varepsilon_2,\varepsilon_1) \right| $',
	# 	x_label=r'Energy $\varepsilon_2$ [eV]',
	# 	y_label=r'Energy $\varepsilon_1$ [eV]',
	# 	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# 	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	# 	magnitude_cmap='turbo',
	# )

    file_path = f'single_output_temp/6mcmc/data_sigma_interp{suffix}.npz'
    data_sigma_interp = np.load(file_path)

    abs_plot(
		mat_abs=data_sigma_interp['mat_abs'],
		extent=data_sigma_interp['extent'],
		save_path=f'parameter_scan/pscan_output/data_sigma_interp{suffix}.png',
		show=False,
		title=r'$ \sigma (\varepsilon_2,\varepsilon_1)$',
		x_label=r'Energy $\varepsilon_2$ [eV]',
		y_label=r'Energy $\varepsilon_1$ [eV]',
		# y_ticks=[-3.1,-1.55,0,1.55,3.1],
		# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
		magnitude_cmap='turbo'
	)

	#
	# INFERRED
	#

    posterior_data_raw = np.load(f'single_output_temp/6mcmc/mcmc_posterior{suffix}.npz')
    posterior_data = _rewrite_posterior_keys(posterior_data_raw)
    posterior_data_raw.close()

    plot_posterior(posterior_data,save_path=f'parameter_scan/pscan_output/mcmc_posterior{suffix}.png')

    file_path = f'single_output_temp/6mcmc/rho_inferred{suffix}.npz'
    rho_inferred = np.load(file_path)

    complex_plot(
		mat_complex=rho_inferred['mat_complex'],
		extent=rho_inferred['extent'],
		save_path=f'parameter_scan/pscan_output/rho_inferred{suffix}.png',
		show=False,
		title='Inferred Density Matrix',
		x_label=r'Energy $\varepsilon_2$ [eV]',
		y_label=r'Energy $\varepsilon_1$ [eV]',
		# y_ticks=[-3.1,-1.55,0,1.55,3.1],
		# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
		magnitude_cmap='turbo',
		phase_cmap='twilight_shifted',
		caption = f'F = {rho_inferred['RES']:.3f}'
	)

	# abs_plot(
	# 	mat_abs=np.abs(rho_inferred['mat_complex']),
	# 	extent=rho_inferred['extent'],
	# 	save_path='plot_output/6mcmc/rho_inferred_abs.png',
	# 	show=False,
	# 	title='Projected Density Matrix',
	# 	x_label=r'Energy $\varepsilon_2$ [eV]',
	# 	y_label=r'Energy $\varepsilon_1$ [eV]',
	# 	# y_ticks=[-3.1,-1.55,0,1.55,3.1],
	# 	# y_tick_labels=[r'$-2\omega_r$',r'$-1\omega_r$',r'$0$',r'$1\omega_r$',r'$2\omega_r$'],
	# 	magnitude_cmap='turbo',
	# 	caption = f'F = {rho_inferred['RES']:.3f}'
	# )