# %% import and definition
import os
import warnings

import cvxpy as cp
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import xarray as xr
from scipy.sparse import dia_matrix
from scipy.spatial.distance import cdist

from routine.cnmf import compute_trace, update_temporal_block
from routine.minian_functions import open_minian
from routine.simulation import generate_data
from routine.utilities import norm, rechunk_like

INT_PATH = "./intermediate/temporal_simulation"
FIG_PATH = "./figs/temporal_simulation"

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

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
subset = minian_ds.coords["unit_id"]
Y, A, C_true, S_true = minian_ds["Y"], minian_ds["A"], minian_ds["C"], minian_ds["S"]
A, C_true, S_true = (
    A.sel(unit_id=subset),
    C_true.sel(unit_id=subset),
    S_true.sel(unit_id=subset),
)
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
    C_true,
)
YrA_tr = compute_trace(Y, A, b, C_true, f).compute()
YrA, gs, tns = update_temporal_block(
    np.array(YrA_tr),
    noise_freq=0.1,
    p=1,
    sparse_penal=0.1,
    max_iters=1000,
    zero_thres=1e-9,
    return_param=True,
)
sps_penal = 10
max_iters = 50
C_new = []
S_new = []
b_new = []
C_bin_new = []
S_bin_new = []
b_bin_new = []
scales = []
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
    C_new.append(c.value)
    S_new.append(s.value)
    b_new.append(b.value)
    # bin prob
    scale = np.ptp(s.value)
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
            cp.norm(y - scale * c_bin - b_bin) + sps_penal * tn * cp.norm(s_bin)
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
                np.linalg.norm(y / scale - G_inv @ s_thres - b_bin.value / scale)
                * scale
                # + sps_penal * tn * np.linalg.norm(s_thres, 1)
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
        scale_new = (y[idx] / est[idx]).item()
        if np.abs(scale_new - scale) <= tol:
            break
        else:
            scale = scale_new
            niter += 1
    else:
        warnings.warn("max scale iteration reached")
    C_bin_new.append(G_inv @ opt_s)
    S_bin_new.append(opt_s)
    b_bin_new.append(b_bin.value)
    scales.append(scale)
# save variables
C_new = xr.DataArray(
    np.concatenate(C_new, axis=1), dims=C_true.dims, coords=C_true.coords, name="C_new"
)
S_new = xr.DataArray(
    np.concatenate(S_new, axis=1), dims=S_true.dims, coords=S_true.coords, name="S_new"
)
b_new = xr.DataArray(
    np.array(b_new),
    dims="unit_id",
    coords={"unit_id": C_true.coords["unit_id"]},
    name="b_new",
)
C_bin_new = xr.DataArray(
    np.concatenate(C_bin_new, axis=1),
    dims=C_true.dims,
    coords=C_true.coords,
    name="C_bin_new",
)
S_bin_new = xr.DataArray(
    np.concatenate(S_bin_new, axis=1),
    dims=S_true.dims,
    coords=S_true.coords,
    name="S_bin_new",
)
b_bin_new = xr.DataArray(
    np.array(b_bin_new),
    dims="unit_id",
    coords={"unit_id": C_true.coords["unit_id"]},
    name="b_bin_new",
)
scales = xr.DataArray(
    np.array(scales),
    dims="unit_id",
    coords={"unit_id": C_true.coords["unit_id"]},
    name="scales",
)
updt_ds = xr.merge(
    [
        C_new.rename("C_new"),
        S_new.rename("S_new"),
        b_new.rename("b_new"),
        C_bin_new.rename("C_bin_new"),
        S_bin_new.rename("S_bin_new"),
        b_bin_new.rename("b_bin_new"),
        scales.rename("scales"),
        YrA_tr.rename("YrA"),
    ]
)
updt_ds.to_netcdf(os.path.join(INT_PATH, "temp_res.nc"))


# %% plot example and metrics
def compute_dist(trueS, newS, metric):
    if metric == "edit":
        dist = np.array(
            [
                Levenshtein.distance(
                    np.array(trueS.sel(unit_id=uid)), np.array(newS.sel(unit_id=uid))
                )
                for uid in trueS.coords["unit_id"]
            ]
        )
    else:
        dist = np.diag(
            cdist(
                trueS.transpose("unit_id", "frame"),
                newS.transpose("unit_id", "frame"),
                metric=metric,
            )
        )
    return pd.DataFrame(
        {
            "method": newS.name,
            "metric": metric,
            "unit_id": trueS.coords["unit_id"],
            "dist": dist,
        }
    )


def norm_per_cell(S):
    return xr.apply_ufunc(
        norm,
        S.astype(float).compute(),
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
    )


temp_ds = xr.open_dataset(os.path.join(INT_PATH, "temp_res.nc"))
true_ds = open_minian(os.path.join(INT_PATH, "simulated")).isel(
    unit_id=temp_ds.coords["unit_id"]
)
Strue = true_ds["S"]
Snew = temp_ds["S_new"]
Smax = Snew.max()
S_ls = [Snew.rename("real_val"), temp_ds["S_bin_new"].rename("bin")]
for thres in np.linspace(0.1, 0.9, 9):
    S_ls.append((Snew > Smax * thres).rename("thres-{:.1f}".format(thres)))
corrs = pd.concat([compute_dist(Strue, ss, "correlation") for ss in S_ls])
hammings = pd.concat([compute_dist(Strue, ss, "hamming") for ss in S_ls[1:]])
edits = pd.concat([compute_dist(Strue, ss, "edit") for ss in S_ls[1:]])
dat = pd.concat([corrs, hammings, edits], ignore_index=True)
g = sns.FacetGrid(dat, row="metric", sharey=False, aspect=3, hue="method")
g.map_dataframe(sns.violinplot, x="method", y="dist", saturation=0.6)
# g.map_dataframe(sns.swarmplot, x="method", y="dist", edgecolor="auto", linewidth=1)
g.figure.savefig(
    os.path.join(FIG_PATH, "bin_metrics.svg"), dpi=500, bbox_inches="tight"
)
nsamp = 10
exp_set = np.random.choice(subset, nsamp, replace=False)
plt_dat = pd.concat(
    [
        norm_per_cell(S_true.sel(unit_id=exp_set)).rename("S_true").to_dataframe(),
        norm_per_cell(temp_ds.sel(unit_id=exp_set)[["YrA"]]).to_dataframe(),
    ]
    + [norm_per_cell(ss.sel(unit_id=exp_set)).to_dataframe() for ss in S_ls]
).reset_index()
plt_dat = plt_dat.melt(id_vars=["frame", "unit_id"])
fig = px.line(plt_dat, facet_row="unit_id", x="frame", y="value", color="variable")
fig.update_layout(height=nsamp * 150)
fig.write_html(os.path.join(FIG_PATH, "bin_example.html"))
