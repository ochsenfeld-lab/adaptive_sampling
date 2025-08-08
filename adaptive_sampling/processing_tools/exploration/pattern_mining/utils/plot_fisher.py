import sys, ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from matplotlib.colors import TwoSlopeNorm, Normalize
import json


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13
})

maxpval = 40

with open(sys.argv[1], 'r') as file:
    data = json.load(file)

freq_A = np.zeros(len(data))
freq_B = np.zeros(len(data))

sign_A = []
sign_B = []

for i, pattern in enumerate(data):
    freq_A[i] = int(pattern[1])
    freq_B[i] = int(pattern[2])
    if pattern[3]:
        sign_A.append(freq_A[i])
        sign_B.append(freq_B[i])
sign_A = np.array(sign_A)
sign_B = np.array(sign_B)



# total number of observations (per condition)
Ta = int(sys.argv[2])
Tb = int(sys.argv[3])
#maxplot = int(max(np.max(freq_A), np.max(freq_B))) + 4
maxplot = int(sys.argv[4])
# Define a grid spanning the range of your frequencies for the background.
# For a smooth image, use a 2D meshgrid with a decent resolution.
grid_A = np.linspace(-1, maxplot, maxplot+1)
grid_B = np.linspace(-1, maxplot, maxplot+1)
xx, yy = np.meshgrid(grid_A, grid_B)

# Create an array to hold the transformed p-values.
# We will compute for each (a, b) pair from the grid.
z = np.zeros_like(xx, dtype=float)

# Loop over the grid; for each cell compute a 2x2 contingency table and run Fisher's exact test.
# Then compute: sign(a - b)*(-log10(p))
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        a = int(round(xx[i, j]))
        b = int(round(yy[i, j]))
        if a < 0: a = 0
        if b < 0: b = 0
        if a > Ta: a = Ta 
        if b > Tb: b = Tb
        
        table = np.array([[a, b],
                            [Ta - a, Tb - b]])
        # fisher_exact returns an odds ratio and a two-tailed p-value.
        oddsratio, p_value = fisher_exact(table)
        # Compute the signed -log10(p_value).
        #z[i, j] = np.sign(a - b) * min(-np.log10(p_value), maxpval)
        z[i, j] = np.sign(a - b) * np.log10(1+min(-np.log10(p_value), maxpval))


# Create a diverging normalization that centers at 0. Values of 0 will show as white.
norm = TwoSlopeNorm(vmin=np.min(z)*1.1, vcenter=0, vmax=np.max(z)*1.1)

# Begin plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the background using pcolormesh.
# Using 'RdBu_r' makes negative values (blue) on one side, positive (red) on the other, white in the middle.
c = ax.pcolormesh(xx, yy, z, cmap='RdBu_r', norm=norm, shading='gouraud')


# Overlay your scatter plot.
ax.scatter(freq_A, freq_B, c='gray', s=8, linewidth=0, marker='.')
ax.scatter(sign_A, sign_B, c='black', s=35, linewidth=0.5, edgecolor='white', marker='*')
ax.set_xlabel(r'Frequency at $T=62K$')
ax.set_ylabel(r'Frequency at $T=39K$')
ax.set_xlim( [-0.9, maxplot] )
ax.set_ylim( [-0.9, maxplot] )
if '_water' in sys.argv[1]:
    ax.set_title(r'Fisher Exact Test Significance - CO$_2$/ NH$_3$/ H$_2$O')
else:    
    ax.set_title(r'Fisher Exact Test Significance - CO$_2$/ NH$_3$')

cbar = fig.colorbar(c, ax=ax, label=r'$-\log_{10}($p-value$)$')

original_ticks = [20, 10, 5, 0, 5, 10, 20]
log_ticks = [-np.log10(21), -np.log10(11), -np.log10(6), 0, np.log10(6), np.log10(11), np.log10(21)]
cbar.set_ticks(log_ticks)
cbar.set_ticklabels([str(t) for t in original_ticks])  # Show original values


plt.savefig(sys.argv[1].split('.')[0]+'.pdf')
