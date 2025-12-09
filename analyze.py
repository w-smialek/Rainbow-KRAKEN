import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter


sqerrs = np.load('scans/1dscans/sqerrs.npy')
fid1s = np.load('scans/1dscans/fid1s.npy')
fid2s = np.load('scans/1dscans/fid2s.npy')
fid3s = np.load('scans/1dscans/fid3s.npy')
fid4s = np.load('scans/1dscans/fid4s.npy')

plt.plot(sqerrs)
plt.show()
plt.plot(fid1s)
plt.show()
plt.plot(fid2s)
plt.show()
plt.plot(fid3s)
plt.show()
plt.plot(fid4s)
plt.show()


sqerrs = median_filter(np.log10(np.load('scans/sqerrs.npy')[:-1,:]),size=(3,3),mode='nearest')
fid1s = median_filter(np.log10(1 - np.load('scans/fid1s.npy')[:-1,:]),size=(3,3),mode='nearest')
fid2s = median_filter(np.log10(1 - np.load('scans/fid2s.npy')[:-1,:]),size=(3,3),mode='nearest')
fid3s = median_filter(np.log10(1 - np.load('scans/fid3s.npy')[:-1,:]),size=(3,3),mode='nearest')
fid4s = median_filter(np.log10(1 - np.load('scans/fid4s.npy')[:-1,:]),size=(3,3),mode='nearest')

# alpha_range = np.array([0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64])
# N_T_range = np.array([250,300,400,500,700,900,1100])

alpha_range = np.linspace(0.01,0.49,25)
N_T_range = np.linspace(250,950,15).astype(int)

# Create meshgrids for 3D surface plots
ALPHA, N_T = np.meshgrid(alpha_range, N_T_range)

# 3D surface plot for sqerrs
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ALPHA, N_T, sqerrs, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_xlabel('Alpha (α)')
ax.set_ylabel('N_T')
ax.set_zlabel('Square Error')
ax.set_title('Square Errors vs Alpha and N_T')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()

# 3D surface plot for fid1s
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ALPHA, N_T, fid1s, cmap='plasma', edgecolor='none', alpha=0.8)
ax.set_xlabel('Alpha (α)')
ax.set_ylabel('N_T')
ax.set_zlabel('Fidelity 1')
ax.set_title('Fidelity 1 vs Alpha and N_T')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()

# 3D surface plot for fid2s
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ALPHA, N_T, fid2s, cmap='plasma', edgecolor='none', alpha=0.8)
ax.set_xlabel('Alpha (α)')
ax.set_ylabel('N_T')
ax.set_zlabel('Fidelity 2')
ax.set_title('Fidelity 2 vs Alpha and N_T')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()

# 3D surface plot for fid3s
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ALPHA, N_T, fid3s, cmap='plasma', edgecolor='none', alpha=0.8)
ax.set_xlabel('Alpha (α)')
ax.set_ylabel('N_T')
ax.set_zlabel('Fidelity 3')
ax.set_title('Fidelity 3 vs Alpha and N_T')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()

# 3D surface plot for fid4s
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ALPHA, N_T, fid4s, cmap='plasma', edgecolor='none', alpha=0.8)
ax.set_xlabel('Alpha (α)')
ax.set_ylabel('N_T')
ax.set_zlabel('Fidelity 4')
ax.set_title('Fidelity 4 vs Alpha and N_T')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()
