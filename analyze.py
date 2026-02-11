import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter


# sqerrs = np.load('scans/1dscans/sqerrs80.npy')
# fid1s = np.load('scans/1dscans/fid1s80.npy')
# fid2s = np.load('scans/1dscans/fid2s80.npy')
# fid3s = np.load('scans/1dscans/fid3s80.npy')
# fid4s = np.load('scans/1dscans/fid4s80.npy')

# plt.plot(sqerrs)
# plt.show()
# plt.plot(fid1s)
# plt.show()
# plt.plot(fid2s)
# plt.show()
# plt.plot(fid3s)
# plt.show()
# plt.plot(fid4s)
# plt.show()


# sqerrs = median_filter(np.log10(np.load('scans/sqerrs.npy')[:-1,:]),size=(3,3),mode='nearest')
# fid1s = median_filter(np.log10(1 - np.load('scans/fid1s.npy')[:-1,:]),size=(3,3),mode='nearest')
# fid2s = median_filter(np.log10(1 - np.load('scans/fid2s.npy')[:-1,:]),size=(3,3),mode='nearest')
# fid3s = median_filter(np.log10(1 - np.load('scans/fid3s.npy')[:-1,:]),size=(3,3),mode='nearest')
# fid4s = median_filter(np.log10(1 - np.load('scans/fid4s.npy')[:-1,:]),size=(3,3),mode='nearest')

# sqerrs = median_filter(np.load('scans/sqerrs.npy')[:-1,:],size=(3,3),mode='nearest')
fid1s = np.flip(median_filter(np.load('scans/newscan/fid1s400.npy')[:,:],size=(3,3),mode='nearest'),axis=0)
fid2s = np.flip(median_filter(np.load('scans/newscan/fid2s400.npy')[:,:],size=(3,3),mode='nearest'),axis=0)
fid3s = np.flip(median_filter(np.load('scans/newscan/fid3s400.npy')[:,:],size=(3,3),mode='nearest'),axis=0)
fid4s = np.flip(median_filter(np.load('scans/newscan/fid4s400.npy')[:,:],size=(3,3),mode='nearest'),axis=0)

# alpha_range = np.array([0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64])
# N_T_range = np.array([250,300,400,500,700,900,1100])

N_T_range = np.linspace(200,1000,10)
alpha_range = np.linspace(10000*4/10,10000*4/2,10)

# Create meshgrids for 3D surface plots
ALPHA, N_T = np.meshgrid(alpha_range, N_T_range)

ALPHA = np.sqrt(np.load('scans/newscan/PSBC.npy'))

# # 3D surface plot for sqerrs
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(ALPHA, N_T, sqerrs, cmap='viridis', edgecolor='none', alpha=0.8)
# ax.set_xlabel('Peak row cnts')
# ax.set_ylabel('N_T')
# ax.set_zlabel('Square Error')
# ax.set_title('Square Errors vs prc and N_T')
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
# plt.tight_layout()
# plt.savefig('scans/sqerrs.png')
# plt.close()

# 3D surface plot for fid1s
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ALPHA, N_T, fid1s, cmap='plasma', edgecolor='none', alpha=0.8)
ax.view_init(elev=30, azim=225)
ax.set_xlabel('Peak row cnts')
ax.set_ylabel('N_T')
ax.set_zlabel('1 - Fidelity 1')
ax.set_title('Fidelity 1 vs prc and N_T')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.savefig('scans/fid1s.png')
plt.close()

# 3D surface plot for fid2s
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ALPHA, N_T, fid2s, cmap='plasma', edgecolor='none', alpha=0.8)
ax.view_init(elev=30, azim=225)
ax.set_xlabel('Peak row cnts')
ax.set_ylabel('N_T')
ax.set_zlabel('1 - Fidelity 2')
ax.set_title('Fidelity 2 vs prc and N_T')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.savefig('scans/fid2s.png')
plt.close()

# 3D surface plot for fid3s
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ALPHA, N_T, fid3s, cmap='plasma', edgecolor='none', alpha=0.8)
ax.view_init(elev=30, azim=225)
ax.set_xlabel('Peak SNR', fontsize=16, fontweight='bold')
ax.set_ylabel('No. of delay steps', fontsize=16, fontweight='bold')
ax.set_zlabel('Quantum state fidelity', fontsize=16, fontweight='bold')
ax.set_title('Retrieval results - coherent peaks', fontsize=20, fontweight='bold')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.savefig('scans/fid3s.png')
plt.close()

# 3D surface plot for fid4s
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ALPHA, N_T, fid4s, cmap='plasma', edgecolor='none', alpha=0.8)
ax.view_init(elev=30, azim=225)
ax.set_xlabel('Peak row cnts')
ax.set_ylabel('N_T')
ax.set_zlabel('1 - Fidelity 4')
ax.set_title('Fidelity 4 vs prc and N_T')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.savefig('scans/fid4s.png')
plt.close()
