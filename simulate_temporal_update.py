# %% import and definition
import os

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.sparse import dia_matrix

from routine.cnmf import compute_trace, update_temporal_block, update_temporal_cvxpy
from routine.minian_functions import open_minian
from routine.simulation import generate_data
from routine.utilities import rechunk_like

INT_PATH = "./intermediate/temporal_simulation"
FIG_PATH = "./figs/temporal_simulation"

os.makedirs(INT_PATH, exist_ok=True)

# %% generate data
Y, A, C, S, shifts = generate_data(
    dpath=INT_PATH,
    ncell=100,
    dims={"height": 256, "width": 256, "frame": 2000},
    sig_scale=1,
    sz_mean=3,
    sz_sigma=0.6,
    sz_min=0.1,
    tmp_pfire=0.01,
    tmp_tau_d=6,
    tmp_tau_r=1,
    bg_nsrc=0,
    bg_tmp_var=0,
    bg_cons_fac=0,
    bg_smth_var=0,
    mo_stp_var=0,
    mo_cons_fac=0,
    post_offset=1,
    post_gain=50,
    save_Y=True,
)

# %% temporal update
minian_ds = open_minian(os.path.join(INT_PATH, "simulated"))
Y, A, C = minian_ds["Y"], minian_ds["A"], minian_ds["C"]
b = rechunk_like(
    xr.DataArray(
        np.zeros((A.sizes["height"], A.sizes["width"])),
        dims=["height", "width"],
        coords={d: A.coords[d] for d in ["height", "width"]},
    ),
    A,
)
f = rechunk_like(
    xr.DataArray(
        np.zeros((Y.sizes["frame"])),
        dims=["frame"],
        coords={"frame": Y.coords["frame"]},
    ),
    C,
)
YrA = compute_trace(Y, A, b, C, f).compute()
c, s, b, c0, g = update_temporal_block(
    np.array(YrA),
    noise_freq=0.1,
    p=2,
    sparse_penal=0.1,
    max_iters=1000,
    zero_thres=1e-9,
)

# %% new method experiment
YrA, gs, tns = update_temporal_block(
    np.array(YrA),
    noise_freq=0.1,
    p=1,
    sparse_penal=0.1,
    max_iters=1000,
    zero_thres=1e-9,
    return_param=True,
)
sps_penal = 10
max_iters = 1000
for y, g, tn in zip(YrA, gs, tns):
    # parameters
    T = len(y)
    G = dia_matrix(
        (
            np.tile(np.concatenate(([1], -g)), (T, 1)).T,
            -np.arange(len(g) + 1),
        ),
        shape=(T, T),
    ).todense()
    y = y - y.min()
    y = y / y.max()
    y = y.reshape((-1, 1))
    G_inv = np.linalg.inv(G)
    # org prob
    c = cp.Variable((T, 1))
    s = cp.Variable((T, 1))
    b = cp.Variable()
    obj = cp.Minimize(cp.norm(y - c - b) + sps_penal * tn * cp.norm(s))
    cons = [s == G @ c, c >= 0, s >= 0, b >= 0]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.ECOS)
    # bin prob
    scale = 1
    niter = 0
    tol = 1e-6
    scale_vals = []
    opt_s_vals = []
    opt_obj_vals = []
    opt_lb_vals = []
    while niter < max_iters:
        c_bin = cp.Variable((T, 1))
        s_bin = cp.Variable((T, 1))
        b_bin = cp.Variable()
        obj = cp.Minimize(
            cp.norm(scale * y - c_bin - b_bin) + sps_penal * tn * cp.norm(s_bin)
        )
        cons = [s_bin == G @ c_bin, c_bin >= 0, b_bin >= 0, s_bin >= 0, s_bin <= 1]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.ECOS)
        svals = []
        objvals = []
        for thres in np.linspace(s_bin.value.min(), s_bin.value.max(), 1000):
            s_thres = s_bin.value >= thres
            svals.append(s_thres)
            objvals.append(
                np.linalg.norm(scale * y - G_inv @ s_thres - b_bin.value)
                + sps_penal * tn * np.linalg.norm(s_thres, 1)
            )
        opt_idx = np.argmin(objvals)
        opt_s = svals[opt_idx]
        scale_vals.append(scale)
        opt_s_vals.append(opt_s)
        opt_obj_vals.append(objvals[opt_idx])
        opt_lb_vals.append(prob.value)
        # scale_new = np.linalg.lstsq(
        #     y.reshape((-1, 1)), np.array(G_inv @ opt_s + b_bin.value).squeeze()
        # )[0][0].squeeze()
        est = G_inv @ opt_s + b_bin.value
        idx = np.argmax(est)
        scale_new = (est[idx] / y[idx]).item()
        if np.abs(scale_new - scale) <= tol:
            break
        else:
            scale = scale_new
            niter += 1
    break

# cons_bin = cons + [s_bin <= 1]
# prob_bin = cp.Problem(obj, cons_bin)
# prob.solve(solver=cp.ECOS)
# %%
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(y.squeeze() * scale_new, label="y")
ax.plot(np.array(G_inv @ opt_s).squeeze(), label="c")
ax.plot(opt_s.squeeze(), label="s")
ax.plot(np.array(G_inv @ opt_s + b_bin.value).squeeze(), label="est")
ax.legend()
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(y.squeeze(), label="y")
ax.plot(np.array(G_inv @ s.value).squeeze(), label="c")
ax.plot(s.value.squeeze(), label="s")
ax.plot(np.array(G_inv @ s.value + b.value).squeeze(), label="est")
ax.legend()
fig, ax = plt.subplots()
ax.plot(scale_vals, label="scale")
ax.plot(opt_obj_vals, label="obj")
ax.plot(opt_lb_vals, label="lb")
ax.legend()
