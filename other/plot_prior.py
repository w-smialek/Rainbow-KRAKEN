import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib import ticker
from matplotlib import font_manager as fm
from pathlib import Path
import re

font_path1 = Path(__file__).resolve().parent / 'fonts' / 'cmu.sans-serif-bold.ttf'
font_path2 = Path(__file__).resolve().parent / 'fonts' / 'cmu.sans-serif-demi-condensed-demicondensed.ttf'
fm.fontManager.addfont(str(font_path1))
fm.fontManager.addfont(str(font_path2))
cmu_sans_bold = fm.FontProperties(fname=str(font_path1)).get_name()
cmu_sans = fm.FontProperties(fname=str(font_path2)).get_name()
plt.rcParams['font.family'] = [cmu_sans_bold, 'DejaVu Sans']
# plt.rcParams['font.family'] = cmu_sans
# plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rc('axes', unicode_minus=False)



def plot_posterior(posterior_panels, save_path=None):
	if not isinstance(posterior_panels, dict):
		posterior_panels = dict(posterior_panels)

	color_hist = "#231696"  # 'tab:blue'

	panel_keys = [key for key in posterior_panels.keys() if not key.startswith('v')]
	n_panels = len(panel_keys)
	n_cols = 4
	n_rows = int(np.ceil(n_panels / n_cols))
	fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.8 * n_rows))
	axs = np.atleast_1d(axs).ravel()

	for ax, label in zip(axs, panel_keys):
		values = posterior_panels[label]
		variable = posterior_panels['v' + label]
		# if 'arg' in label:
		#     values_wrapped = ((values + np.pi) % (2.0 * np.pi)) - np.pi
		#     ax.hist(values_wrapped, bins=np.linspace(-np.pi, np.pi, 41), density=False, alpha=0.75, color=color_hist)
		#     ax.axvline(_circular_mean(values_wrapped), color='black', linestyle='--', linewidth=1.0)
		#     ax.set_xlim(-np.pi, np.pi)
		# else:
		#     ax.hist(values, bins=40, density=False, alpha=0.75, color=color_hist)
		#     ax.axvline(np.mean(values), color='black', linestyle='--', linewidth=1.0)
		ax.plot(variable, values)

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

	fig.suptitle('Prior Distributions PDF of Fitted Parameters', fontsize=30, y=1.02, weight='bold')
	plt.tight_layout()
	if save_path is not None:
		plt.savefig(save_path, dpi=200, bbox_inches='tight')
	plt.close()


def normal(x, mu, s):
	return 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / s) ** 2)


def _exp(r,mu,s):
	variable = np.linspace(0.01, r, 100)
	values_new = normal(np.log(variable), mu, s)
	return variable, values_new

def _normal(r,mu,s):
	variable = np.linspace(-r, r, 100)
	values_new = normal(variable,mu,s)
	return variable, values_new

def _sigmoid(mu,s):
	variable = np.linspace(0.01, 0.99, 100)
	values_new = normal(-np.log(1/variable-1), mu, s)
	return variable, values_new

def _wrapped(mu,s):
	variable = np.linspace(-np.pi, np.pi, 100)
	values_new = normal(variable, mu, s)
	for k in range(-3,4):
		values_new += normal(variable, mu + 2*k*np.pi, s)
	return variable, values_new

renamed = {}

variable, values_new = _exp(15,np.log(1.5),1)
renamed['$C_{ii}$'] = values_new
renamed['v$C_{ii}$'] = variable

variable, values_new = _sigmoid(0.0,1.3)
renamed['$\\mu_{i}$'] = values_new
renamed['v$\\mu_{i}$'] = 2*variable + 24.0

variable, values_new = _exp(0.4,np.log(0.1),0.35)
renamed['$\\sigma_{i}$'] = values_new
renamed['v$\\sigma_{i}$'] = variable

variable, values_new = _normal(10,0.0,3.0)
renamed['$\\beta_{i}$'] = values_new
renamed['v$\\beta_{i}$'] = variable

variable, values_new = _normal(5,0.0,1.5)
renamed['$\\tau_{i}$'] = values_new
renamed['v$\\tau_{i}$'] = variable

variable, values_new = _exp(20,np.log(2.0),1.0)
renamed['$\\gamma_{i}$'] = values_new
renamed['v$\\gamma_{i}$'] = variable

variable, values_new = _sigmoid(-2.0,1.0)
renamed['$|C_{ij}| / \\sqrt{ C_{ii} C_{jj} } \\text{ non-coherent}$'] = values_new
renamed['v$|C_{ij}| / \\sqrt{ C_{ii} C_{jj} } \\text{ non-coherent}$'] = variable

variable, values_new = _sigmoid(0.0,1.0)
renamed['$|C_{ij}| / \\sqrt{ C_{ii} C_{jj} } \\text{ coherent}$'] = values_new
renamed['v$|C_{ij}| / \\sqrt{ C_{ii} C_{jj} } \\text{ coherent}$'] = variable

variable, values_new = _wrapped(0.0,1.0)
renamed['$\\text{arg}(C_{ij})$'] = values_new
renamed['v$\\text{arg}(C_{ij})$'] = variable

variable, values_new = _sigmoid(0.0,1.0)
renamed['$\\eta_{ij}$'] = values_new
renamed['v$\\eta_{ij}$'] = variable*2 - 1

plot_posterior(renamed,save_path='bb.png')