# %% imports and definition
import os

from routine.cnmf import compute_trace, update_temporal_cvxpy
from routine.simulation import generate_data

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
    bg_nsrc=100,
    bg_tmp_var=2,
    bg_cons_fac=0.1,
    bg_smth_var=60,
    mo_stp_var=0,
    mo_cons_fac=0,
    post_offset=1,
    post_gain=50,
)
