import matplotlib.pyplot as plt
import numpy as np

sqerrs = np.load('scans/sqerrs.npy')
fid1s = 1 - np.load('scans/fid1s.npy')
fid2s = 1 - np.load('scans/fid2s.npy')
fid3s = 1 - np.load('scans/fid3s.npy')
fid4s = 1 - np.load('scans/fid4s.npy')

alpha_range = np.array([0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64])
N_T_range = np.array([250,300,400,500,700,900,1100])

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
