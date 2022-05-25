import sys
import numpy as np
from free_energy_tools import *


def P1(x):
    a = 8.0e-6
    d = 80.0
    e = 160.0
    s1 = (x - d) * (x - d)
    s2 = (x - e) * (x - e)
    return a * s1 * s2


print("reading trajectory...\n")
traj = np.loadtxt("CV_traj.dat", skiprows=1)
traj = traj[:2000000]

print("setting up grid...\n")
sigma = 2.0
windows = np.arange(70.0, 170.0, 2.0)
t = traj[:, 0]
cv = traj[:, 1]  # reaction coordinate
la = traj[:, 2]  # extended system
W_list = []
index_list = []
pmf_list = []
cv_list = []
idx = np.arange(0, len(traj))
print("Starting MBAR...")
sys.stdout.flush()
for i in range(100):

    choice = np.random.choice(idx, len(traj), replace=True)
    cv_i = cv[choice]
    la_i = la[choice]

    traj_list_i, indices_i, meta_f_i = get_us_windows(
        windows, cv_i, la_i, sigma, T=300.0
    )
    index_list.append(indices_i)
    cv_list.append(traj_list_i)
    print(
        "Minimum Samples:\t", min([len(traj) for traj in traj_list_i if len(traj) > 0])
    )
    W_i = MBAR(
        traj_list_i,
        meta_f_i,
        max_iter=10000,
        T=300,
        conv=1.0e-7,
        outfreq=500,
        autocorr=False,
    )
    pmf_mbar, _ = mbar_pmf(windows, traj_list_i, W_i, T=300)
    print(pmf_mbar)
    pmf_list.append(pmf_mbar)
    W_list.append(W_i)
    np.savez(
        "weights_mbar",
        W=W_list,
        index=index_list,
        cv=cv_list,
        pmf=pmf_list,
        allow_pickle=True,
    )
    sys.stdout.flush()
