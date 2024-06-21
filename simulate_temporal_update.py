# %% import and definition
import itertools as itt
import os
import warnings

import cv2
import cvxpy as cp
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.sparse as sps
import seaborn as sns
import xarray as xr
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from routine.cnmf import compute_trace, update_temporal_block
from routine.minian_functions import open_minian
from routine.simulation import generate_data, tau2AR
from routine.utilities import norm, rechunk_like

INT_PATH = "./intermediate/temporal_simulation"
FIG_PATH = "./figs/temporal_simulation"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


def thresS(S, nthres, rename=True):
    Smax = S.max()
    if rename:
        return [
            (S > Smax * th).rename(S.name + "-th_{:.1f}".format(th))
            for th in np.linspace(0.1, 0.9, nthres)
        ]
    else:
        return [(S > Smax * th) for th in np.linspace(0.1, 0.9, nthres)]


# %% generate data
Y, A, C, S, shifts, C_gt, S_gt = generate_data(
    dpath=INT_PATH,
    ncell=100,
    upsample=PARAM_UPSAMP,
    dims={"height": 256, "width": 256, "frame": 2000},
    sig_scale=1,
    sz_mean=3,
    sz_sigma=0.6,
    sz_min=0.1,
    tmp_pfire=0.003,
    tmp_tau_d=PARAM_TAU_D,
    tmp_tau_r=PARAM_TAU_R,
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
minian_ds = open_minian(os.path.join(INT_PATH, "simulated"), return_dict=True)
subset = minian_ds["A"].coords["unit_id"]
Y, A, C_gt, S_gt, C_gt_true, S_gt_true = (
    minian_ds["Y"],
    minian_ds["A"],
    minian_ds["C"],
    minian_ds["S"],
    minian_ds["C_true"],
    minian_ds["S_true"],
)
A, C_gt, S_gt = (
    A.sel(unit_id=subset),
    C_gt.sel(unit_id=subset),
    S_gt.sel(unit_id=subset),
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
    C_gt,
)
YrA = compute_trace(Y, A, b, C_gt, f).compute()
updt_ds = [YrA.rename("YrA")]
for up_type, up_factor in {"org": 1, "upsamp": PARAM_UPSAMP}.items():
    _, _, tns = update_temporal_block(
        np.array(YrA),
        noise_freq=0.1,
        p=2,
        add_lag=100,
        sparse_penal=0.1,
        max_iters=1000,
        zero_thres=1e-9,
        return_param=True,
    )
    gs = np.tile(
        tau2AR(PARAM_TAU_D * up_factor, PARAM_TAU_R * up_factor),
        (A.sizes["unit_id"], 1),
    )
    sps_penal = 10
    max_iters = 50
    res = {"C": [], "S": [], "b": [], "C-bin": [], "S-bin": [], "b-bin": [], "scal": []}
    for y, g, tn in tqdm(zip(np.array(YrA), gs, tns), total=np.array(YrA).shape[0]):
        # parameters
        Torg = len(y)
        T = Torg * up_factor
        G = sps.dia_matrix(
            (
                np.tile(np.concatenate(([1], -g)), (T, 1)).T,
                -np.arange(len(g) + 1),
            ),
            shape=(T, T),
        ).tocsc()
        G_inv = sps.linalg.inv(G)
        rs_vec = np.zeros(T)
        rs_vec[:up_factor] = 1
        Rs = sps.coo_matrix(
            np.stack([np.roll(rs_vec, up_factor * i) for i in range(Torg)], axis=0)
        )
        RG = (Rs @ G_inv).todense()
        y = y - y.min()
        y = y / y.max()
        y = y.reshape((-1, 1))
        # org prob
        c = cp.Variable((T, 1))
        s = cp.Variable((T, 1))
        b = cp.Variable()
        obj = cp.Minimize(cp.norm(y - Rs @ c - b) + sps_penal * tn * cp.norm(s))
        cons = [s == G @ c, c >= 0, s >= 0, b >= 0]
        prob = cp.Problem(obj, cons)
        prob.solve()
        res["C"].append(c.value)
        res["S"].append(s.value)
        res["b"].append(b.value)
        # no sparse prob
        c_init = cp.Variable((T, 1))
        s_init = cp.Variable((T, 1))
        b_init = cp.Variable()
        obj_init = cp.Minimize(cp.norm(y - Rs @ c_init - b_init))
        cons_init = [s_init == G @ c_init, c_init >= 0, s_init >= 0, b_init >= 0]
        prob_init = cp.Problem(obj_init, cons_init)
        prob_init.solve()
        # bin prob
        scale = np.ptp(s_init.value)
        niter = 0
        tol = 1e-6
        s_bin_df = pd.DataFrame(
            {"s_bin": s.value.squeeze(), "frame": np.arange(T), "iter": -1}
        )
        scale_df = None
        opt_s_df = None
        obj_df = None
        lb_df = None
        while niter < max_iters:
            c_bin = cp.Variable((T, 1))
            s_bin = cp.Variable((T, 1))
            b_bin = cp.Variable()
            obj = cp.Minimize(
                cp.norm(y - scale * Rs @ c_bin - b_bin)
                # + sps_penal * tn * cp.norm(s_bin)
            )
            cons = [s_bin == G @ c_bin, c_bin >= 0, b_bin >= 0, s_bin >= 0, s_bin <= 1]
            prob = cp.Problem(obj, cons)
            prob.solve()
            objvals = []
            svals = thresS(s_bin.value, 1000, rename=False)
            objvals = [
                np.linalg.norm(y - scale * (RG @ ss) - b_bin.value) for ss in svals
            ]
            opt_idx = np.argmin(objvals)
            opt_s = svals[opt_idx]
            opt_obj = objvals[opt_idx]
            try:
                opt_obj_last = obj_df["obj"].min()
            except TypeError:
                opt_obj_last = np.inf
            scale_df = pd.concat(
                [scale_df, pd.DataFrame([{"scale": scale, "iter": niter}])]
            )
            s_bin_df = pd.concat(
                [
                    s_bin_df,
                    pd.DataFrame(
                        {
                            "s_bin": s_bin.value.squeeze(),
                            "frame": np.arange(T),
                            "iter": niter,
                        }
                    ),
                ]
            )
            opt_s_df = pd.concat(
                [
                    opt_s_df,
                    pd.DataFrame(
                        {"opt_s": opt_s.squeeze(), "frame": np.arange(T), "iter": niter}
                    ),
                ]
            )
            obj_df = pd.concat(
                [obj_df, pd.DataFrame([{"obj": opt_obj, "iter": niter}])]
            )
            lb_df = pd.concat(
                [lb_df, pd.DataFrame([{"lb": prob.value, "iter": niter}])]
            )
            scale_new = np.linalg.lstsq(
                RG @ opt_s, (y - b_bin.value).squeeze(), rcond=None
            )[0].item()
            # est = G_inv @ opt_s + b_bin.value
            # idx = np.argmax(est)
            # scale_new = (y[idx] / est[idx]).item()
            if np.abs(scale_new - scale) <= tol:
                break
            elif opt_obj_last - opt_obj >= 0 and opt_obj_last - opt_obj <= tol:
                break
            else:
                scale = scale_new
                niter += 1
        else:
            warnings.warn("max scale iteration reached")
        res["C-bin"].append(G_inv @ opt_s)
        res["S-bin"].append(opt_s)
        res["b-bin"].append(b_bin.value)
        res["scal"].append(scale)
    # save variables
    for vname, dat in res.items():
        if vname.startswith("b") or vname.startswith("scal"):
            updt_ds.append(
                xr.DataArray(
                    np.array(dat),
                    dims="unit_id",
                    coords={"unit_id": A.coords["unit_id"]},
                    name="-".join([vname, up_type]),
                )
            )
        else:
            updt_ds.append(
                xr.DataArray(
                    np.concatenate(dat, axis=1),
                    dims=["frame", "unit_id"],
                    coords={
                        "frame": (
                            C_gt_true.coords["frame"]
                            if up_type == "upsamp"
                            else YrA.coords["frame"]
                        ),
                        "unit_id": A.coords["unit_id"],
                    },
                    name="-".join([vname, up_type]),
                )
            )
updt_ds = xr.merge(updt_ds)
updt_ds.to_netcdf(os.path.join(INT_PATH, "temp_res.nc"))


# %% plot example and metrics
def dilate1d(a, kernel):
    return cv2.dilate(a.astype(float), kernel).squeeze()


def compute_dist(trueS, newS, metric, corr_dilation=1):
    if metric == "correlation" and corr_dilation:
        kn = np.ones(2 * corr_dilation + 1)
        trueS = xr.apply_ufunc(
            dilate1d,
            trueS.compute(),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            kwargs={"kernel": kn},
        )
        newS = xr.apply_ufunc(
            dilate1d,
            newS.compute(),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            kwargs={"kernel": kn},
        )
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
    Sname = newS.name.split("-")
    if "org" in Sname:
        mthd = "original"
        Sname.remove("org")
    elif "upsamp" in Sname:
        mthd = "upsampled"
        Sname.remove("upsamp")
    elif "updn" in Sname:
        mthd = "up/down"
        Sname.remove("updn")
    else:
        mthd = "unknown"
    return pd.DataFrame(
        {
            "variable": "-".join(Sname),
            "method": mthd,
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


def compute_metrics(S, S_true, mets, nthres: int = None):
    S, S_true = S.dropna("frame"), S_true.dropna("frame")
    if nthres is not None:
        S_ls = thresS(S, nthres)
    else:
        S_ls = [S]
    res_ls = [compute_dist(S_true, curS, met) for curS, met in itt.product(S_ls, mets)]
    return pd.concat(res_ls)


updt_ds = xr.open_dataset(os.path.join(INT_PATH, "temp_res.nc"))
true_ds = open_minian(os.path.join(INT_PATH, "simulated")).isel(
    unit_id=updt_ds.coords["unit_id"]
)
subset = updt_ds.coords["unit_id"]
S_gt, S_gt_true = true_ds["S"].dropna("frame", how="all"), true_ds["S_true"]
S_org, S_bin_org, S_up, S_bin_up, YrA = (
    updt_ds["S-org"].dropna("frame", how="all"),
    updt_ds["S-bin-org"].dropna("frame", how="all"),
    updt_ds["S-upsamp"],
    updt_ds["S-bin-upsamp"],
    updt_ds["YrA"].dropna("frame", how="all"),
)
S_updn, S_bin_updn = (
    S_up.coarsen({"frame": 10}).sum().rename("S-updn"),
    S_bin_up.coarsen({"frame": 10}).sum().rename("S-bin-updn"),
)
S_updn = S_updn.assign_coords({"frame": np.ceil(S_updn.coords["frame"]).astype(int)})
S_bin_updn = S_bin_updn.assign_coords(
    {"frame": np.ceil(S_bin_updn.coords["frame"]).astype(int)}
)
met_ds = [
    (S_org, S_gt, {"mets": ["correlation"]}),
    (S_org, S_gt, {"mets": ["correlation", "hamming", "edit"], "nthres": 9}),
    (S_bin_org, S_gt, {"mets": ["correlation", "hamming", "edit"]}),
    (S_up, S_gt_true, {"mets": ["correlation"]}),
    (S_up, S_gt_true, {"mets": ["correlation", "hamming", "edit"], "nthres": 9}),
    (S_bin_up, S_gt_true, {"mets": ["correlation", "hamming", "edit"]}),
    (S_updn, S_gt, {"mets": ["correlation"]}),
    (S_updn, S_gt, {"mets": ["correlation", "hamming", "edit"], "nthres": 9}),
    (S_bin_updn, S_gt, {"mets": ["correlation", "hamming", "edit"]}),
]
met_res = pd.concat(
    [compute_metrics(m[0], m[1], **m[2]) for m in met_ds], ignore_index=True
)
sns.set_theme(style="darkgrid")
g = sns.FacetGrid(
    met_res,
    row="metric",
    col="method",
    sharey="row",
    sharex="col",
    aspect=2,
    hue="variable",
    margin_titles=True,
)
g.map_dataframe(
    sns.violinplot,
    x="variable",
    y="dist",
    bw_adjust=0.5,
    cut=0.3,
    saturation=0.6,
    log_scale=True,
)
# g.map_dataframe(sns.swarmplot, x="variable", y="dist", edgecolor="auto", linewidth=1)
g.tick_params(axis="x", rotation=90)
g.figure.savefig(os.path.join(FIG_PATH, "metrics.svg"), dpi=500, bbox_inches="tight")
nsamp = min(10, len(subset))
exp_set = np.random.choice(subset, nsamp, replace=False)
fig_dict = {
    "exp_original": [S_gt, YrA, S_org, S_bin_org] + thresS(S_org, 9),
    # "exp_upsamp": [S_gt_true, YrA_interp, S_up, S_bin_up] + thresS(S_up, 9),
    "exp_updn": [S_gt, YrA, S_updn, S_bin_updn] + thresS(S_updn, 9),
}
for figname, plt_trs in fig_dict.items():
    plt_dat = pd.concat(
        [norm_per_cell(tr.sel(unit_id=exp_set)).to_dataframe() for tr in plt_trs]
    ).reset_index()
    plt_dat = plt_dat.melt(id_vars=["frame", "unit_id"])
    fig = px.line(plt_dat, facet_row="unit_id", x="frame", y="value", color="variable")
    fig.update_layout(height=nsamp * 150)
    fig.write_html(os.path.join(FIG_PATH, "{}.html".format(figname)))
