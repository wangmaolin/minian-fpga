# %% imports and definition
import os

import numpy as np
import xarray as xr

from routine.cnmf import compute_trace, update_temporal_block
from routine.minian_functions import open_minian
from routine.simulation import generate_data
from routine.utilities import rechunk_like

INT_PATH = "./intermediate/temporal_simulation"
FIG_PATH = "./figs/temporal_simulation"

os.makedirs(INT_PATH, exist_ok=True)

# %% generate simulated data
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
