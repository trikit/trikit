import numpy as np
import pandas as pd
import os
import sys; sys.path.append("G:\\Repos\\trikit\\trikit")
import collections
import datetime
import pathlib
import os.path
import warnings
import scipy
import trikit
from trikit import triangle
import pickle
import importlib
# %load_ext line_profiler




pd.options.display.float_format = '{:.5f}'.format
np.set_printoptions(suppress=True, precision=5)
pd.options.mode.chained_assignment = None
np.set_printoptions(linewidth=1000)
np.core.arrayprint._line_width = 1000
pd.set_option("display.max_columns", 10000)
pd.set_option("display.width", 10000)



pp = "G:\\Repos\\trikit\\trikit\\datasets\\RAA.csv"
df = pd.read_csv(pp)


# importlib.reload(trikit)

tri = triangle.CumTriangle(data=df, origin="origin", dev="dev", value="value")
bootargs = {"sims":500, "procdist":"gamma", "parametric":False, "q":[.75, .95]}
bcl = tri.cl(range_method="bootstrap", **bootargs)
# bcl.plot()


tri = triangle.CumTriangle(data=df, origin="origin", dev="dev", value="value")
# mcl = tri.cl(range_method="mack")

from trikit.chainladder import mack
mcl = mack.MackChainLadder(cumtri=tri)









# sys.path.append("G:/Repos")
# from bs import bs
# %lprun -T lprof0 -f bs bs()



# BootstrapChainLadder --------------------------------------------------------


ldfs = tri.cl().ldfs
fitted_tri_cum = tri.copy(deep=True)
for ii in range(fitted_tri_cum.shape[0]):
    iterrow = fitted_tri_cum.iloc[ii, :]
    if iterrow.isnull().any():
        # Find first NaN element in iterrow.
        nan_hdr = iterrow.isnull()[iterrow.isnull()==True].index[0]
        nan_idx = fitted_tri_cum.columns.tolist().index(nan_hdr)
        init_idx = nan_idx - 1
    else:
        # If here, iterrow is the most mature exposure period.
        init_idx = fitted_tri_cum.shape[1] - 1

    # Set to NaN any development periods earlier than init_idx.
    fitted_tri_cum.iloc[ii, :init_idx] = np.NaN

    # Iterate over rows, undeveloping triangle from latest diagonal.
    for j in range(fitted_tri_cum.iloc[ii, :init_idx].size, 0, -1):
        prev_col_idx, curr_col_idx, curr_ldf_idx = j, j - 1, j - 1
        prev_col_val = fitted_tri_cum.iloc[ii, prev_col_idx]
        curr_ldf_val = ldfs.iloc[curr_ldf_idx]
        fitted_tri_cum.iloc[ii, curr_col_idx] = (prev_col_val / curr_ldf_val)



fitted_tri_incr = fitted_tri_cum.diff(axis=1)
fitted_tri_incr.iloc[:, 0] = fitted_tri_cum.iloc[:, 0]

I = tri.to_incr()
m = fitted_tri_incr
resid_us = (I - m) / np.sqrt(m.abs())

dof_ = tri.nbr_cells - (tri.columns.size - 1) + tri.index.size
resid_adj = np.sqrt(tri.nbr_cells / dof_) * resid_us

resid_ = resid_adj.iloc[:-1,:-1].values.ravel()
sampling_dist = resid_[np.logical_and(~np.isnan(resid_), resid_!=0)]



# _bs_samples

random_state = 516
neg_handler="first"
sims = 1000
parametric = False

from numpy.random import RandomState
if random_state is not None:
    if isinstance(random_state, int):
        prng = RandomState(random_state)
    elif isinstance(random_state, RandomState):
        prng = random_state
else:
    prng = RandomState()

sampling_dist = sampling_dist.flatten()

fti = fitted_tri_incr.reset_index(drop=False).rename({"index":"origin"}, axis=1)
dfm = pd.melt(fti, id_vars=["origin"], var_name="dev", value_name="value")
dfm = dfm[~np.isnan(dfm["value"])].astype(
    {"origin":np.int_, "dev":np.int_, "value":np.float_}
    )

# Handle first period negative cells as specified by `neg_handler`.
if np.any(dfm["value"]<0):

    if neg_handler=="first":
        dfm["value"] = np.where(
            np.logical_and(dfm["dev"].values==1, dfm["value"].values<0),
            1., dfm["value"].values
        )

    elif neg_handler=="all":
        # Obtain reference to minimum triangle cell value, then
        # add the absolute value of that amount plus one to all
        # other triangle cells.
        add2cells = np.abs(dfm["value"].min()) + 1
        dfm["value"] = dfm["value"] + add2cells

    else:
        raise ValueError("`neg_handler` must be in ['first', 'all'].")


dfi = tri.to_tbl(drop_nas=False).drop("value", axis=1)
dfp = dfi.merge(dfm, how="outer", on=["origin", "dev"])
dfp["rectype"] = np.where(np.isnan(dfp["value"].values), "forecast", "actual")
dfp = dfp.rename({"value":"incr"}, axis=1)
dfp["incr_sqrt"] = np.sqrt(dfp["incr"].values)
dfrtypes = {"origin":np.int, "dev":np.int_, "incr":np.float_,
            "incr_sqrt":np.float_, "rectype":np.str,}
dfrcols = ["origin", "dev", "incr", "rectype", "incr_sqrt"]
dfr = pd.DataFrame(np.tile(dfp, (sims, 1)), columns=dfrcols).astype(dfrtypes)
dfr["sim"] = np.divmod(dfr.index, tri.shape[0] * tri.shape[1])[0]
sample_size = dfr.shape[0]

if parametric:
    # Sample random residuals from normal distribution with zero mean.
    dfr["resid"] = prng.normal(
        loc=0, scale=sampling_dist.std(ddof=1), size=sample_size
        )
else:
    # Sample random residual from adjusted pearson residuals.
    dfr["resid"] = prng.choice(
        sampling_dist, sample_size, replace=True
        )

# Calcuate resampled incremental and cumulative losses.
dfr["resid"] = np.where(dfr["rectype"].values=="forecast", np.NaN, dfr["resid"].values)
dfr = dfr.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True)
dfr["samp_incr"] = dfr["incr"].values + dfr["resid"].values * dfr["incr_sqrt"].values
dfr["samp_cum"]  = dfr.groupby(["sim", "origin"], as_index=False)["samp_incr"].cumsum()
# dfsamples = dfr.reset_index(drop=True)

# samp_path = "G:/Repos/Temp/dfsamples.pkl"
# with open(samp_path, 'wb') as f:
#     pickle.dump(dfsamples, f, pickle.HIGHEST_PROTOCOL)

# dfsamples
# samp_path = "G:/Repos/Temp/dfsamples.pkl"
# with open(samp_path, 'wb') as f:
#     pickle.dump(dfsamples, f, pickle.HIGHEST_PROTOCOL)
samp_path = "G:/Repos/Temp/dfsamples.pkl"
with open(samp_path, "rb") as fpkl:
    dfsamples = pickle.load(fpkl)

# _bs_ldfs.
ldfs_path = "G:/Repos/Temp/dfldfs1.pkl"
with open(ldfs_path, "rb") as fpkl:
    bs_ldfs = pickle.load(fpkl)#.rename({"ldf":"ldf0"}, axis=1)

# bs_forecasts ----------------------------------------------------------------

dflvi = tri.rlvi.reset_index(drop=False)
dflvi = dflvi.rename({"index":"origin", "dev":"l_act_dev"}, axis=1)
dflvi = dflvi.drop("col_offset", axis=1)
scale_param = (resid_us**2).sum().sum() / dof_
dfcombined = dfsamples.merge(bs_ldfs, on=["sim", "dev"], how="left")
dfcombined = dfcombined.merge(dflvi, on=["origin"], how="left")
dfcombined = dfcombined.reset_index(drop=True).sort_values(by=["sim", "origin", "dev"])


min_origin_year = dfcombined["origin"].values.min()
dfcombined["_l_init_indx"] = np.where(
    dfcombined["dev"].values>=dfcombined["l_act_dev"].values, dfcombined.index.values, -1)
dfacts = dfcombined[(dfcombined["origin"].values==min_origin_year) | (dfcombined["_l_init_indx"].values==-1)]
dffcst = dfcombined[~dfcombined.index.isin(dfacts.index)].sort_values(by=["sim", "origin", "dev"])
dffcst["_l_act_indx"] = dffcst.groupby(["sim", "origin"])["_l_init_indx"].transform("min")
dffcst["l_act_cum"] = dffcst.lookup(dffcst["_l_act_indx"].values, ["samp_cum"] * dffcst.shape[0])
dffcst["_cum_ldf"] = dffcst.groupby(["sim", "origin"])["ldf"].transform("cumprod").shift(periods=1)
dffcst["_samp_cum2"] = dffcst["l_act_cum"].values * dffcst["_cum_ldf"].values
dffcst["_samp_cum2"] = np.where(np.isnan(dffcst["_samp_cum2"].values), 0, dffcst["_samp_cum2"].values)
dffcst["cum_final"] = np.where(np.isnan(dffcst["samp_cum"].values), 0, dffcst["samp_cum"].values) + dffcst["_samp_cum2"].values

# Combine forecasts with actuals then compute incremental losses by sim and origin.
dffcst = dffcst.drop(labels=["samp_cum", "samp_incr"], axis=1).rename(columns={"cum_final":"samp_cum"})
dfsqrd = pd.concat([dffcst, dfacts], sort=True).sort_values(by=["sim", "origin", "dev"])
dfsqrd["_dev1_ind"] = (dfsqrd["dev"].values==1) * 1
dfsqrd["_incr_dev1"] = dfsqrd["_dev1_ind"].values * dfsqrd["samp_cum"].values
dfsqrd["_incr_dev2"] = dfsqrd.groupby(["sim", "origin"])["samp_cum"].diff(periods=1)
dfsqrd["_incr_dev2"] = np.where(np.isnan(dfsqrd["_incr_dev2"].values), 0, dfsqrd["_incr_dev2"].values)
dfsqrd["samp_incr"] = dfsqrd["_incr_dev1"].values + dfsqrd["_incr_dev2"].values
dfsqrd["var"] = np.abs(dfsqrd["samp_incr"].values * scale_param)
dfsqrd["sign"] = np.where(dfsqrd["samp_incr"].values > 0, 1, -1)
dfsqrd = dfsqrd.drop(labels=[i for i in dfsqrd.columns if i.startswith("_")], axis=1)

dfsqrd.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True)


min_origin_year = dfcombined["origin"].values.min()
dfcombined["_l_init_indx"] = np.where(
    dfcombined["dev"].values>=dfcombined["l_act_dev"].values, dfcombined.index.values, -1)
dfacts = dfcombined[(dfcombined["origin"].values==min_origin_year) | (dfcombined["_l_init_indx"].values==-1)]
dffcst = dfcombined[~dfcombined.index.isin(dfacts.index)].sort_values(by=["sim", "origin", "dev"])
dffcst["_l_act_indx"] = dffcst.groupby(["sim", "origin"])["_l_init_indx"].transform("min")
dffcst["l_act_cum"] = dffcst.lookup(dffcst["_l_act_indx"].values, ["samp_cum"] * dffcst.shape[0])
dffcst["_cum_ldf"] = dffcst.groupby(["sim", "origin"])["ldf"].transform("cumprod").shift(periods=1)
dffcst["_samp_cum2"] = np.nan_to_num((dffcst["l_act_cum"].values * dffcst["_cum_ldf"].values), 0)
dffcst["cum_final"] = np.nan_to_num(dffcst["samp_cum"].values, 0) + dffcst["_samp_cum2"].values

# Combine forecasts with actuals then compute incremental losses by sim and origin.
dffcst = dffcst.drop(labels=["samp_cum", "samp_incr"], axis=1).rename(columns={"cum_final":"samp_cum"})
dfsqrd = pd.concat([dffcst, dfacts], sort=True).sort_values(by=["sim", "origin", "dev"])
dfsqrd["_incr_dev1"] = np.nan_to_num(np.where(dfsqrd["dev"].values==1, dfsqrd["samp_cum"].values, np.NaN), 0)


dfsqrd["_incr_dev2"] = np.nan_to_num(dfsqrd.groupby(["sim", "origin"])["samp_cum"].diff(periods=1), 0)




dfsqrd["samp_incr"] = dfsqrd["_incr_dev1"].values + dfsqrd["_incr_dev2"].values
dfsqrd["var"] = np.abs(dfsqrd["samp_incr"].values * scale_param)
dfsqrd["sign"] = np.where(dfsqrd["samp_incr"].values > 0, 1, -1)
dfsqrd = dfsqrd.drop(labels=[i for i in dfsqrd.columns if i.startswith("_")], axis=1)

dfforecasts = dfsqrd.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True)
dfforecasts["param2"] = scale_param
dfforecasts["param1"] = np.abs(dfforecasts["samp_incr"].values / dfforecasts["param2"].values)
def fdist(param1, param2):
    """gamma.rvs(a=param1, scale=param2, size=1, random_state=None)"""
    return(prng.gamma(param1, param2))

dfforecasts["final_incr"] = np.where(
    dfforecasts["rectype"].values=="forecast",
    fdist(dfforecasts["param1"].values, dfforecasts["param2"].values) * dfforecasts["sign"].values,
    dfforecasts["samp_incr"].values
    )
dfforecasts["final_cum"] = dfforecasts.groupby(["sim", "origin"])["final_incr"].cumsum()
dfforecasts = dfforecasts.rename({"final_cum":"ultimate", "l_act_cum":"latest"}, axis=1)
dfprocerror = dfforecasts.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True)

keepcols = ["sim", "origin", "latest", "ultimate", "reserve"]
max_devp = dfprocerror["dev"].values.max()
dfprocerror["reserve"] = dfprocerror["ultimate"] - dfprocerror["latest"]
dfreserves = dfprocerror[dfprocerror["dev"].values==max_devp][keepcols].drop_duplicates()
dfreserves["latest"]  = np.where(
    np.isnan(dfreserves["latest"].values),
    dfreserves["ultimate"].values, dfreserves["latest"].values
    )
dfreserves["reserve"] = np.nan_to_num(dfreserves["reserve"].values, 0)
dfreserves = dfreserves.sort_values(by=["origin", "sim"]).reset_index(drop=True)


