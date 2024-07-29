# %% import and definition
import itertools as itt
import os
import warnings

import cv2
import cvxpy as cp
import Levenshtein
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.sparse as sps
import seaborn as sns
import xarray as xr
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from routine.bin_algo import (
    construct_G,
    construct_R,
    estimate_coefs,
    max_thres,
    solve_deconv,
    solve_deconv_bin,
)
from routine.minian_functions import open_minian
from routine.simulation import AR2tau, generate_data, tau2AR
from routine.utilities import norm

IN_PATH = "./intermediate/temporal_simulation/"
INT_PATH = "./intermediate/ar_update"
FIG_PATH = "./figs/ar_update"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10
PARAM_EST_AR = True

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# %% AR update exp approach
minian_ds = open_minian(os.path.join(IN_PATH, "simulated"), return_dict=True)
Y, A, C_gt, S_gt, C_gt_true, S_gt_true = (
    minian_ds["Y"],
    minian_ds["A"],
    minian_ds["C"],
    minian_ds["S"],
    minian_ds["C_true"],
    minian_ds["S_true"],
)
c, s = np.array(C_gt.isel(unit_id=0)).reshape((-1, 1)), np.array(
    S_gt.isel(unit_id=0)
).reshape((-1, 1))
y = c
tau1_init, tau2_init = 7, 1.2

T = y.shape[0]
lam1, lam2 = cp.Variable(), cp.Variable()
# pulse_kernel = cp.exp(-np.arange(T) * lam1) - cp.exp(-np.arange(T) * 1 / tau2_init)
pulse_kernel = cp.exp(-np.arange(T) * lam1)

M0 = pulse_kernel.reshape((-1, 1))
M1n = [
    cp.vstack([np.zeros(i).reshape((-1, 1)), pulse_kernel[:-i].reshape((-1, 1))])
    for i in range(1, T)
]
M = cp.hstack([M0] + M1n)
obj = cp.norm(M @ s - y)


# %% AR update reverse solve approach
def convolve_g(s, g):
    G = construct_G(g, len(s))
    Gi = sps.linalg.inv(G)
    return np.array(Gi @ s.reshape((-1, 1))).squeeze()


def solve_g(y, s, norm="l2", masking=False):
    T = len(s)
    theta_1, theta_2 = cp.Variable(), cp.Variable()
    G = (
        np.eye(T)
        + np.diag(-np.ones(T - 1), -1) * theta_1
        + np.diag(-np.ones(T - 2), -2) * theta_2
    )
    if masking:
        idx = np.where(s)[0]
        M = np.zeros((len(idx), T))
        for i, j in enumerate(idx):
            M[i, j] = 1
    else:
        M = np.eye(T)
    if norm == "l2":
        obj = cp.Minimize(cp.norm(M @ (G @ y - s)))
    elif norm == "l1":
        obj = cp.Minimize(cp.norm(M @ (G @ y - s), 1))
    cons = [theta_1 >= 0, theta_2 <= 0]
    prob = cp.Problem(obj, cons)
    prob.solve()
    return theta_1.value, theta_2.value


def solve_g_cons(y, s, lam_tol=1e-6, lam_start=1, max_iter=30):
    T = len(s)
    i_iter = 0
    lam = lam_start
    lam_last = lam_start
    ch_last = -np.inf
    while i_iter < max_iter:
        theta_1, theta_2 = cp.Variable(), cp.Variable()
        G = (
            np.eye(T)
            + np.diag(-np.ones(T - 1), -1) * theta_1
            + np.diag(-np.ones(T - 2), -2) * theta_2
        )
        obj = cp.Minimize(cp.norm(G @ y - s) + lam * (-theta_2 - theta_1))
        cons = [theta_1 >= 0, theta_2 <= 0]
        prob = cp.Problem(obj, cons)
        prob.solve()
        th1, th2 = theta_1.value, theta_2.value
        ch_root = th1**2 + 4 * th2
        if ch_root > 0:
            lam_new = lam / 2
        else:
            if ch_last > 0:
                lam_new = lam + (lam_last - lam) / 2
            else:
                lam_new = lam * 2
        if (lam - lam_new) >= 0 and (lam - lam_new) <= lam_tol:
            break
        else:
            i_iter += 1
            lam_last = lam
            lam = lam_new
            ch_last = ch_root
            print(
                "th1: {}, th2: {}, ch: {}, lam: {}".format(th1, th2, ch_root, lam_last)
            )
    else:
        warnings.warn("max lam iteration reached")
    return th1, th2


minian_ds = open_minian(os.path.join(IN_PATH, "simulated"), return_dict=True)
Y, A, C_gt, S_gt, C_gt_true, S_gt_true = (
    minian_ds["Y"],
    minian_ds["A"],
    minian_ds["C"],
    minian_ds["S"],
    minian_ds["C_true"],
    minian_ds["S_true"],
)
c, s = np.array(C_gt.isel(unit_id=0)).reshape((-1, 1)), np.array(
    S_gt.isel(unit_id=0)
).reshape((-1, 1))
noise_lev = [0, 1, 2, 5, 10]
methods = [
    "est-smth0",
    "est-smth20",
    "est-smth100",
    "est-naive",
    "solve-l1",
    "solve-l2",
]
res_df = []
for ns in noise_lev:
    y = c + ns * np.random.random(c.shape)
    res_df.append(
        pd.DataFrame(
            {
                "method": "y",
                "noise": ns,
                "frame": np.arange(len(y)),
                "value": norm(y.squeeze()),
            }
        )
    )
    res_df.append(
        pd.DataFrame(
            {
                "method": "c",
                "noise": ns,
                "frame": np.arange(len(y)),
                "value": norm(c.squeeze()),
            }
        )
    )
    res_df.append(
        pd.DataFrame(
            {
                "method": "s",
                "noise": ns,
                "frame": np.arange(len(y)),
                "value": norm(s.squeeze()),
            }
        )
    )
    for mthd in methods:
        m, param = mthd.split("-")
        if m == "est":
            if param.startswith("smth"):
                g, tn = estimate_coefs(
                    y, p=2, noise_freq=0.1, use_smooth=True, add_lag=int(param[4:])
                )
            else:
                g, tn = estimate_coefs(
                    y, p=2, noise_freq=0.5, use_smooth=False, add_lag=0
                )
        elif m == "solve":
            g = solve_g(y, s, norm=param)
            G = construct_G(g, len(y))
            y_est = (G @ y).squeeze()
            res_df.append(
                pd.DataFrame(
                    {
                        "method": mthd + "-y",
                        "noise": ns,
                        "frame": np.arange(len(y)),
                        "value": norm(y_est),
                    }
                )
            )
        c_est = convolve_g(s, g)
        print("method: {}, g: {}".format(mthd, g))
        res_df.append(
            pd.DataFrame(
                {
                    "method": mthd,
                    "noise": ns,
                    "frame": np.arange(len(y)),
                    "value": norm(c_est),
                }
            )
        )
res_df = pd.concat(res_df, ignore_index=True)
fig = px.line(
    res_df, facet_row="noise", x="frame", y="value", color="method", line_group="noise"
)
fig.write_html(os.path.join(FIG_PATH, "AR_update.html"))
