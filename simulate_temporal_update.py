# %% import and definition
import itertools as itt
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
from tqdm.auto import tqdm

from routine.cnmf import compute_trace, update_temporal_block
from routine.minian_functions import open_minian
from routine.simulation import generate_data
from routine.utilities import norm, rechunk_like

INT_PATH = "./intermediate/temporal_simulation"
FIG_PATH = "./figs/temporal_simulation"

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# %% generate data
Y, A, C, S, shifts, C_gt, S_gt = generate_data(
    dpath=INT_PATH,
    ncell=100,
    upsample=10,
    dims={"height": 256, "width": 256, "frame": 2000},
    sig_scale=1,
    sz_mean=3,
    sz_sigma=0.6,
    sz_min=0.1,
    tmp_pfire=0.003,
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
minian_ds = open_minian(os.path.join(INT_PATH, "simulated"), return_dict=True)
subset = minian_ds["A"].coords["unit_id"][:5]
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
YrA_interp = YrA.interp(
    {"frame": C_gt_true.coords["frame"]}, kwargs={"fill_value": "extrapolate"}
)
updt_ds = [YrA.rename("YrA"), YrA_interp.rename("YrA_interp")]
for up_type, cur_YrA in {"org": YrA, "upsamp": YrA_interp}.items():
    cur_YrA_ps, gs, tns = update_temporal_block(
        np.array(cur_YrA),
        noise_freq=0.1,
        p=2,
        add_lag=100,
        sparse_penal=0.1,
        max_iters=1000,
        zero_thres=1e-9,
        return_param=True,
    )
    sps_penal = 10
    max_iters = 50
    res = {"C": [], "S": [], "b": [], "C-bin": [], "S-bin": [], "b-bin": [], "scal": []}
    for y, g, tn in tqdm(
        zip(np.array(cur_YrA), gs, tns), total=np.array(cur_YrA).shape[0]
    ):
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
        prob.solve()
        res["C"].append(c.value)
        res["S"].append(s.value)
        res["b"].append(b.value)
        # no sparse prob
        c_init = cp.Variable((T, 1))
        s_init = cp.Variable((T, 1))
        b_init = cp.Variable()
        obj_init = cp.Minimize(cp.norm(y - c_init - b_init))
        cons_init = [s_init == G @ c_init, c_init >= 0, s_init >= 0, b_init >= 0]
        prob_init = cp.Problem(obj_init, cons_init)
        prob_init.solve()
        # bin prob
        scale = np.ptp(s_init.value)
        niter = 0
        tol = 1e-6
        scale_df = []
        s_bin_df = []
        opt_s_df = []
        obj_df = []
        lb_df = []
        while niter < max_iters:
            c_bin = cp.Variable((T, 1))
            s_bin = cp.Variable((T, 1))
            b_bin = cp.Variable()
            obj = cp.Minimize(
                cp.norm(y - scale * c_bin - b_bin)
                # + sps_penal * tn * cp.norm(s_bin)
            )
            cons = [s_bin == G @ c_bin, c_bin >= 0, b_bin >= 0, s_bin >= 0, s_bin <= 1]
            prob = cp.Problem(obj, cons)
            prob.solve()
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
            scale_df.append(pd.DataFrame([{"scale": scale, "iter": niter}]))
            s_bin_df.append(
                pd.DataFrame(
                    {
                        "s_bin": s_bin.value.squeeze(),
                        "frame": np.arange(T),
                        "iter": niter,
                    }
                )
            )
            opt_s_df.append(
                pd.DataFrame(
                    {"opt_s": opt_s.squeeze(), "frame": np.arange(T), "iter": niter}
                )
            )
            obj_df.append(pd.DataFrame([{"obj": objvals[opt_idx], "iter": niter}]))
            lb_df.append(pd.DataFrame([{"lb": prob.value, "iter": niter}]))
            scale_new = np.linalg.lstsq(
                G_inv @ opt_s, (y - b_bin.value).squeeze(), rcond=None
            )[0].item()
            # est = G_inv @ opt_s + b_bin.value
            # idx = np.argmax(est)
            # scale_new = (y[idx] / est[idx]).item()
            if np.abs(scale_new - scale) <= tol:
                break
            else:
                scale = scale_new
                niter += 1
        else:
            warnings.warn("max scale iteration reached")
        scale_df = pd.concat(scale_df, ignore_index=True)
        s_bin_df = pd.concat(
            s_bin_df
            + [
                pd.DataFrame(
                    {"s_bin": s.value.squeeze(), "frame": np.arange(T), "iter": -1}
                )
            ],
            ignore_index=True,
        )
        opt_s_df = pd.concat(opt_s_df, ignore_index=True)
        obj_df = pd.concat(obj_df, ignore_index=True)
        lb_df = pd.concat(lb_df, ignore_index=True)
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
                        "frame": cur_YrA.coords["frame"],
                        "unit_id": A.coords["unit_id"],
                    },
                    name="-".join([vname, up_type]),
                )
            )
updt_ds = xr.merge(updt_ds)
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
    Sname = newS.name.split("-")
    if "org" in Sname:
        mthd = "original"
        Sname.remove("org")
    elif "upsamp" in newS.name:
        mthd = "upsampled"
        Sname.remove("upsamp")
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


def thresS(S, nthres):
    Smax = S.max()
    S_ls = [
        (S > Smax * th).rename(S.name + "-th_{:.1f}".format(th))
        for th in np.linspace(0.1, 0.9, nthres)
    ]
    return S_ls


updt_ds = xr.open_dataset(os.path.join(INT_PATH, "temp_res.nc"))
true_ds = open_minian(os.path.join(INT_PATH, "simulated")).isel(
    unit_id=updt_ds.coords["unit_id"]
)

S_gt, S_gt_true = true_ds["S"].dropna("frame", how="all"), true_ds["S_true"]
S_org, S_bin_org, S_up, S_bin_up, YrA, YrA_up = (
    updt_ds["S-org"].dropna("frame", how="all"),
    updt_ds["S-bin-org"].dropna("frame", how="all"),
    updt_ds["S-upsamp"],
    updt_ds["S-bin-upsamp"],
    updt_ds["YrA"].dropna("frame", how="all"),
    updt_ds["YrA_interp"],
)
met_ds = [
    (S_org, S_gt, {"mets": ["correlation"]}),
    (S_org, S_gt, {"mets": ["correlation", "hamming", "edit"], "nthres": 9}),
    (S_bin_org, S_gt, {"mets": ["correlation", "hamming", "edit"]}),
    (S_up, S_gt_true, {"mets": ["correlation"]}),
    (S_up, S_gt_true, {"mets": ["correlation", "hamming", "edit"], "nthres": 9}),
    (S_bin_up, S_gt_true, {"mets": ["correlation", "hamming", "edit"]}),
]
met_res = pd.concat(
    [compute_metrics(m[0], m[1], **m[2]) for m in met_ds], ignore_index=True
)
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
    sns.violinplot, x="variable", y="dist", bw_adjust=0.5, cut=0.3, saturation=0.6
)
g.map_dataframe(sns.swarmplot, x="variable", y="dist", edgecolor="auto", linewidth=1)
g.tick_params(axis="x", rotation=90)
g.figure.savefig(os.path.join(FIG_PATH, "metrics.svg"), dpi=500, bbox_inches="tight")
nsamp = min(10, len(subset))
exp_set = np.random.choice(subset, nsamp, replace=False)
fig_dict = {
    "exp_original": [S_gt, YrA, S_org, S_bin_org] + thresS(S_org, 9),
    # "exp_upsamp": [S_gt_true, YrA_interp, S_up, S_bin_up] + thresS(S_up, 9),
}
for figname, plt_trs in fig_dict.items():
    plt_dat = pd.concat(
        [norm_per_cell(tr.sel(unit_id=exp_set)).to_dataframe() for tr in plt_trs]
    ).reset_index()
    plt_dat = plt_dat.melt(id_vars=["frame", "unit_id"])
    fig = px.line(plt_dat, facet_row="unit_id", x="frame", y="value", color="variable")
    fig.update_layout(height=nsamp * 150)
    fig.write_html(os.path.join(FIG_PATH, "{}.html".format(figname)))
