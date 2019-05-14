import sys; sys.path.insert(0, "E:/Repos/trikit/tests")
from trikit_tests import *


# def _bs_reserves1(pedf):
#
#     grpsdf = pedf.groupby(["sim", "origin"], sort=False)
#
#     dflist = [
#         grpsdf.get_group(i).reset_index(drop=True) for i in grpsdf.groups
#         ]
#
#     for j in range(len(dflist)):
#
#         dflist[j].sort_values(by=["dev"], ascending=True, inplace=True)
#
#         l_act_indx  = dflist[j].index[dflist[j].rectype=="actual"].max()
#         l_fcst_init = dflist[j].index[dflist[j].rectype=="forecast"].max()
#         l_fcst_indx = l_act_indx if np.isnan(l_fcst_init) else l_fcst_init
#         dflist[j].loc[l_fcst_indx, "latest_actual"] = dflist[j].at[l_act_indx, "final_cum"]
#         dflist[j]["reserve"] = dflist[j]["final_cum"] - dflist[j]["latest_actual"]
#     resdf = pd.concat(dflist)
#     return(resdf.sort_values(["sim", "origin", "dev"]).reset_index(drop=True))
#
#
# def _bs_reserves2(pedf):
#     keepcols = ["sim", "origin", "latest", "ultimate", "reserve"]
#     max_devp = pedf["dev"].values.max()
#     pedf = pedf.rename(columns={"final_cum":"ultimate", "l_act_cum":"latest"})
#     pedf["reserve"] = pedf["ultimate"] - pedf["latest"]
#     resvdf = pedf[pedf["dev"].values==max_devp][keepcols].drop_duplicates()
#     #resvdf = resvdf[keepcols].drop_duplicates()
#     resvdf["latest"]  = np.where(np.isnan(resvdf["latest"].values), resvdf["ultimate"].values, resvdf["latest"].values)
#     resvdf["reserve"] = np.nan_to_num(resvdf["reserve"].values, 0)
#     return(resvdf.sort_values(by=["origin", "sim"]).reset_index(drop=True))
#
#
#
#
#
# def _ldfs0(samples):
#
#     df, lvi  = samples, bcl.tri.clvi.reset_index(drop=False)
#
#     lvi.rename(
#         columns={
#             "index":"dev", "origin":"last_origin",
#             "row_offset":"origin_offset"
#             }, inplace=True
#         )
#
#     df = df.merge(lvi, how="left", on=["dev"])
#     dfgrp = df.groupby(["sim"], sort=False)
#     ldflist = list()
#     simlist = [
#         dfgrp.get_group(i).reset_index(drop=True)
#             for i in dfgrp.groups
#         ]
#
#     for indx, simgrp in enumerate(simlist):
#         idf = simgrp[["origin", "dev", "samp_cum", "last_origin"]]
#         idf = idf.reset_index(drop=True)
#         ll  = list()
#         pairs = zip(
#             idf.dev.unique()[::-1][:-1].tolist(),
#             idf.dev.unique()[::-1][1:].tolist()
#             )
#
#         for devp2, devp1 in pairs:
#             devp2sum = idf[idf.dev==devp2].samp_cum.sum()
#             devp2lo  = idf[idf.dev==devp2].last_origin.unique()[0]
#             devp1sum = idf[(idf.dev==devp1) & (idf.origin <= devp2lo)].samp_cum.sum()
#             ll.append((devp1, (devp2sum / devp1sum)))
#
#         idevp, ildfs = zip(*ll)
#         ldflist.append(
#             pd.DataFrame({"dev":idevp, "ldf":ildfs, "sim":indx})
#             )
#
#     _ldfs = pd.concat(ldflist)[["sim", "dev", "ldf"]]
#     return(_ldfs.sort_values(by=["sim", "dev"]).reset_index(drop=True))
#
#
#
# def _ldfs1(samples):
#
#     #samples = _samples
#     lvi = bcl.tri.clvi.reset_index(drop=False)
#
#     lvi.rename(
#         columns={
#             "index":"dev", "origin":"last_origin", "row_offset":"origin_offset"
#             }, inplace=True
#         )
#
#     samples2 = samples.merge(lvi, how="left", on=["dev"])
#     dfgroups = samples2.groupby(["sim"], sort=False)
#     keepcols = ["sim", "origin", "dev", "samp_cum", "last_origin"]
#     ldflist  = []
#
#     for indx, simgrp in dfgroups:
#
#         initdf = simgrp[keepcols].reset_index(drop=True)
#         iterdf = initdf.sort_values(by=["sim", "dev", "origin"], ascending=[True, True, True])
#         iterdf = iterdf[~np.isnan(iterdf.samp_cum)].reset_index(drop=True)
#         iterdf["_aggdev1"]  = iterdf.groupby(["sim", "dev"])["samp_cum"].transform(pd.Series.sum)
#         iterdf["_aggdev2"] = iterdf.apply(lambda rec: 0 if rec.origin==rec.last_origin else rec.samp_cum, axis=1)
#         iterdf["_aggdev2"] = iterdf.groupby(["sim", "dev"])["_aggdev2"].transform(pd.Series.sum)
#         uniqdf = iterdf[["sim", "dev", "_aggdev1", "_aggdev2"]].drop_duplicates().reset_index(drop=True)
#         uniqdf["_aggdev2"] = uniqdf["_aggdev2"].shift(periods=1, axis=0)
#         uniqdf["dev"] = uniqdf["dev"].shift(periods=1, axis=0)
#         uniqdf["ldf"] = uniqdf._aggdev1 / uniqdf._aggdev2
#         ldflist.append(uniqdf)
#
#     _ldfs = pd.concat(ldflist).dropna(how="any").drop(labels=["_aggdev1", "_aggdev2"], axis=1)
#
#     _ldfs["dev"] = _ldfs["dev"].astype(np.integer)
#
#     return(_ldfs.sort_values(["sim", "dev"]).reset_index(drop=True))
#
#
#
# def _ldfs2(samples):
#
#     #samples = _samples
#     keepcols = ["sim", "origin", "dev", "samp_cum", "last_origin"]
#
#     lvi = self.tri.clvi.reset_index(drop=False)
#
#     lvi.rename(
#         columns={
#             "index":"dev", "origin":"last_origin", "row_offset":"origin_offset"
#             }, inplace=True
#         )
#
#     samples2 = samples.merge(lvi, how="left", on=["dev"])
#     initdf = samples2[keepcols].sort_values(by=["sim", "dev", "origin"], ascending=[True, True, True])
#     ldfsdf = initdf[~np.isnan(initdf["samp_cum"])].reset_index(drop=True)
#     ldfsdf["_aggdev1"] = ldfsdf.groupby(["sim", "dev"])["samp_cum"].transform("sum")
#     ldfsdf["_aggdev2"] = np.where(ldfsdf["origin"].values==ldfsdf["last_origin"].values, 0, ldfsdf["samp_cum"].values)
#     ldfsdf["_aggdev2"] = ldfsdf.groupby(["sim", "dev"])["_aggdev2"].transform("sum")
#     uniqdf = ldfsdf[["sim", "dev", "_aggdev1", "_aggdev2"]].drop_duplicates().reset_index(drop=True)
#     uniqdf["_aggdev2"] = uniqdf["_aggdev2"].shift(periods=1, axis=0)
#     uniqdf["dev"] = uniqdf["dev"].shift(periods=1, axis=0)
#     _ldfs = uniqdf[uniqdf["_aggdev2"]!=0].dropna(how="any")
#     _ldfs["ldf"] = _ldfs["_aggdev1"] / _ldfs["_aggdev2"]
#     _ldfs["dev"] = _ldfs["dev"].astype(np.integer)
#     return(_ldfs[["sim", "dev", "ldf"]].reset_index(drop=True))
#
#
# # _bs_forecasts ==============================================================]
#
#
# def _forecasts0(samples, ldfs):
#
#     fulldf = samples.merge(ldfs, how="left", on=["sim", "dev"])
#     grpsdf = fulldf.groupby(["sim", "origin"], sort=False)
#     dflist = [
#         grpsdf.get_group(i).reset_index(drop=True)
#             for i in grpsdf.groups
#         ]
#
#     for j in range(len(dflist)):
#         dflist[j].sort_values(by=["dev"], ascending=True, inplace=True)
#         if np.any(dflist[j].rectype=="forecast"):
#             l_act_indx  = dflist[j].index[dflist[j].rectype=="actual"].max()
#             f_fcst_indx = dflist[j].index[dflist[j].rectype=="forecast"].min()
#             l_act_cum   = dflist[j].loc[l_act_indx, "samp_cum"]
#             dflist[j].loc[f_fcst_indx:, "samp_cum"] = \
#                 dflist[j].loc[l_act_indx:, "ldf"].cumprod().values[:-1] * l_act_cum
#             dflist[j]["samp_incr"][1:] = dflist[j]["samp_cum"].diff(periods=1)[1:]
#     sqrddf = pd.concat(dflist)
#     sqrddf["var"] = \
#         sqrddf.apply(
#             lambda rec: np.abs(rec.samp_incr * 632) \
#                 if rec.rectype=="forecast" else np.NaN, axis=1)
#     sqrddf["sign"] = \
#         sqrddf.apply(
#             lambda rec: 1 if rec.samp_incr > 0 else -1, axis=1)
#     return(sqrddf.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True))
#
#
#
#
# def _forecasts1a(samples, ldfs):
#
#     fulldf = samples.merge(ldfs, how="left", on=["sim", "dev"]).sort_values(by=["sim", "origin", "dev"])
#     fulldf["_fcst_test"] = fulldf.groupby(["sim", "origin"])["rectype"].transform(lambda v: np.any(v.values=="forecast"))
#     actsdf = fulldf[fulldf["_fcst_test"]==False]; df = fulldf[fulldf["_fcst_test"]==True]
#     df["_l_act_indx"]    = df.groupby(["sim", "origin"])["samp_cum"].transform(lambda v: v.last_valid_index())
#     df["_fcst_ind"]      = (df["_l_act_indx"].values <= df.index)*1
#     df["_l_act_cum"]     = df.groupby(["sim", "origin"])["samp_cum"].transform(lambda v: v[v.last_valid_index()])
#     df["_revldf"]        = np.nan_to_num(np.where(df["_fcst_ind"].values==1, df["ldf"].values, 1), 0)
#     df["_cumldf"]        = df.groupby(["sim", "origin"])["_revldf"].transform(lambda v: v.cumprod()).shift(periods=1, axis=0)
#     df["_samp_cum2"]     = np.nan_to_num((df["_l_act_cum"].values * df["_fcst_ind"].values * df["_cumldf"].values), 0)
#     df["samp_cum_final"] = np.nan_to_num(df["samp_cum"].values * (1 - df["_fcst_ind"].values), 0) + \
#                            np.nan_to_num(df["_samp_cum2"].values * df["_fcst_ind"].values, 0)
#
#     # Drop and rename columns.
#     df.drop(labels=["samp_cum", "samp_incr"], axis=1, inplace=True)
#     df.drop(labels=[i for i in df.columns if i.startswith("_")], axis=1, inplace=True)
#     df.rename(columns={"samp_cum_final":"samp_cum"}, inplace=True)
#     df["samp_incr"] = df["samp_cum"]
#     df["samp_incr"][1:] = df["samp_cum"].diff(periods=1)[1:]
#     actsdf.drop(labels="_fcst_test", axis=1, inplace=True)
#     sqrddf = pd.concat([df, actsdf], sort=True)
#     sqrddf["var"]  = np.where(sqrddf["rectype"].values=="forecast", np.abs(sqrddf["samp_incr"].values * 632.3368030912758), np.NaN)
#     sqrddf["sign"] = np.where(sqrddf["samp_incr"].values > 0, 1, -1)
#     return(sqrddf.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True))
#
#
#
#
#
# def _forecasts1b(samples):
#     fulldf = _samples.merge(_ldfs, how="left", on=["sim", "dev"]).sort_values(by=["sim", "origin", "dev"])
#     #fulldf = samples
#     fulldf["_fcst_test"] = fulldf.groupby(["sim", "origin"])["rectype"].transform(lambda v: np.any(v.values=="forecast"))
#     actsdf = fulldf[fulldf["_fcst_test"]==False]
#     df = fulldf[fulldf["_fcst_test"]==True]
#
#     df["_l_act_indx"]    = df.groupby(["sim", "origin"])["samp_cum"].transform(lambda v: v.last_valid_index())
#     df["_fcst_ind"]      = (df["_l_act_indx"].values <= df.index)*1
#     df["_l_act_cum"]     = df.groupby(["sim", "origin"])["samp_cum"].transform(lambda v: v[v.last_valid_index()])
#     df["_revldf"]        = np.nan_to_num(np.where(df["_fcst_ind"].values==1, df["ldf"].values, 1), 0)
#     df["_cumldf"]        = df.groupby(["sim", "origin"])["_revldf"].transform(lambda v: v.cumprod()).shift(periods=1, axis=0)
#     df["_samp_cum2"]     = np.nan_to_num((df["_l_act_cum"].values * df["_fcst_ind"].values * df["_cumldf"].values), 0)
#     df["samp_cum_final"] = np.nan_to_num(df["samp_cum"].values * (1 - df["_fcst_ind"].values), 0) + \
#                            np.nan_to_num(df["_samp_cum2"].values * df["_fcst_ind"].values, 0)
#
#     # Drop and rename columns.
#     df.drop(labels=["samp_cum", "samp_incr"], axis=1, inplace=True)
#     df.drop(labels=[i for i in df.columns if i.startswith("_")], axis=1, inplace=True)
#     df.rename(columns={"samp_cum_final":"samp_cum"}, inplace=True)
#     df["samp_incr"] = df["samp_cum"]
#     df["samp_incr"][1:] = df["samp_cum"].diff(periods=1)[1:]
#     actsdf.drop(labels="_fcst_test", axis=1, inplace=True)
#     sqrddf = pd.concat([df, actsdf], sort=True)
#     sqrddf["var"]  = np.where(sqrddf["rectype"].values=="forecast", np.abs(sqrddf["samp_incr"].values * 632.3368030912758), np.NaN)
#     sqrddf["sign"] = np.where(sqrddf["samp_incr"].values > 0, 1, -1)
#     return(sqrddf.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True))
#
#
# # Alternative approach:
# # Filter out all actual cells prior to performing ultimate calculation
#
#
#
# def _forecasts1c(df):
#
#     #df = combined
#
#     min_origin_year = df["origin"].values.min()
#     df["_l_init_indx"] = np.where(df["dev"].values>=df["_l_act_dev"].values, df.index.values, -1)
#     actsdf = df[(df["origin"].values==min_origin_year) | (df["_l_init_indx"].values==-1)]
#     fcstdf = df[~df.index.isin(actsdf.index)].sort_values(by=["sim", "origin", "dev"])
#     fcstdf["_l_act_indx"] = fcstdf.groupby(["sim", "origin"])["_l_init_indx"].transform("min")
#     fcstdf["_l_act_cum"]  = fcstdf.lookup(fcstdf["_l_act_indx"].values, ["samp_cum"] * fcstdf.shape[0])
#     fcstdf["_cum_ldf"]    = fcstdf.groupby(["sim", "origin"])["ldf"].transform("cumprod").shift(periods=1, axis=0)
#     fcstdf["_samp_cum2"]  = np.nan_to_num((fcstdf["_l_act_cum"].values * fcstdf["_cum_ldf"].values), 0)
#     fcstdf["cum_final"]   = np.nan_to_num(fcstdf["samp_cum"].values, 0) + fcstdf["_samp_cum2"].values
#
#     # Drop and rename columns.
#     fcstdf.drop(labels=["samp_cum", "samp_incr"], axis=1, inplace=True)
#     fcstdf.rename(columns={"cum_final":"samp_cum"}, inplace=True)
#
#     sqrddf = pd.concat([fcstdf, actsdf], sort=True).sort_values(by=["sim", "origin", "dev"])
#     sqrddf["_incr_dev1"] = np.nan_to_num(np.where(sqrddf["dev"].values==1, sqrddf["samp_cum"].values, np.NaN), 0)
#     sqrddf["_incr_dev2"] = np.nan_to_num(sqrddf.groupby(["sim", "origin"])["samp_cum"].diff(periods=1), 0)
#     sqrddf["samp_incr"]  = sqrddf["_incr_dev1"].values + sqrddf["_incr_dev2"].values
#     sqrddf["var"]        = np.abs(sqrddf["samp_incr"].values * 632.3368030912758)
#     sqrddf["sign"]       = np.where(sqrddf["samp_incr"].values > 0, 1, -1)
#     sqrddf.drop(labels=[i for i in sqrddf.columns if i.startswith("_")], axis=1, inplace=True)
#
#     colorder = [
#         "sim", "origin", "dev","incr", "incr_sqrt", "rectype",
#         "resid", "samp_incr", "samp_cum", "ldf", "var", "sign"
#         ]
#
#     return(sqrddf[colorder].sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True))
#
#
#
# # _bs_process_error ==========================================================]
#
# def _bspe1(fcstdf, procdist="gamma", random_state=None):
#
#     #forceasts = fcst2#forecasts
#     procdist = "gamma"
#     random_state = None
#
#     # Initialize pseudo random number generator.
#     if random_state is not None:
#         if isinstance(random_state, int):
#             prng = RandomState(random_state)
#         elif isinstance(random_state, RandomState):
#             prng = random_state
#     else:
#         prng = RandomState()
#
#     # Parameterize distribution to incorporate process variance.
#     if procdist.strip().lower()=="gamma":
#         fcstdf["param2"] = 632.3368030912758
#         fcstdf["param1"] = fcstdf.apply(lambda rec: np.abs(rec.samp_incr / rec.param2) if rec.rectype=="forecast" else np.NaN, axis=1)
#
#         def fdist(param1, param2):
#             """gamma.rvs(a=param1, scale=param2, size=1, random_state=None)"""
#             return(prng.gamma(param1, param2))
#
#     elif procdist.strip().lower()=="odp":
#          # For odp: param1=n (`n`), param2=prob (`p`).
#         fcstdf["param2"] = ((632.3368030912758 - 1) / 632.3368030912758)
#         fcstdf["param1"] = fcstdf.apply(lambda rec: ((rec.samp_incr * (1 - rec.param2)) / rec.param2) if rec.rectype=="forecast" else np.NaN, axis=1)
#
#         def fdist(param1, param2):
#             """nbinom.rvs(n=param1, p=param2, size=1, random_state=None)"""
#             return(prng.nbinom(param1, param2))
#
#     grpsdf = fcstdf.groupby(["sim","origin"], sort=False)
#     dflist = [grpsdf.get_group(i).reset_index(drop=True) for i in grpsdf.groups]
#
#     for j in range(len(dflist)):
#         dflist[j]["final_incr"] = dflist[j].apply(lambda rec: fdist(rec.param1, rec.param2) * rec.sign if rec.rectype=="forecast" else rec.samp_incr, axis=1)
#
#     # Concatenate elements of dflist; compute final cumulative sum.
#     ultdf = pd.concat(dflist).sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True)
#     ultdf["final_cum"] = ultdf.groupby(["sim", "origin"])["final_incr"].cumsum()
#     return(ultdf.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True))
#
#
#
#
#
# def _bspe2(fcstdf, procdist="gamma", random_state=None):
#
#     # Initialize pseudo random number generator.
#     if random_state is not None:
#         if isinstance(random_state, int):
#             prng = RandomState(random_state)
#         elif isinstance(random_state, RandomState):
#             prng = random_state
#     else:
#         prng = RandomState()
#
#     # Parameterize distribution to incorporate process variance.
#     if procdist.strip().lower()=="gamma":
#         fcstdf["param2"] = 632.3368030912758
#         fcstdf["param1"]  = np.abs(fcstdf["samp_incr"].values / fcstdf["param2"].values)
#
#         def fdist(param1, param2):
#             """gamma.rvs(a=param1, scale=param2, size=1, random_state=None)"""
#             return(prng.gamma(param1, param2))
#
#     elif procdist.strip().lower()=="odp":
#          # For odp: param1=n (`n`), param2=prob (`p`).
#         fcstdf["param2"] = ((632.3368030912758 - 1) / 632.3368030912758)
#         fcstdf["param1"] = (fcstdf["samp_incr"].values * (1 - fcstdf["param2"].values)) / fcstdf["param2"].values
#
#         def fdist(param1, param2):
#             """nbinom.rvs(n=param1, p=param2, size=1, random_state=None)"""
#             return(prng.nbinom(param1, param2))
#
#     fcstdf["final_incr"] = np.where(
#         fcstdf["rectype"].values=="forecast",
#         fdist(fcstdf["param1"].values, fcstdf["param2"].values) * fcstdf["sign"].values,
#         fcstdf["samp_incr"].values
#         )
#
#     fcstdf["final_cum"]  = fcstdf.groupby(["sim", "origin"])["final_incr"].cumsum()
#
#     return(fcstdf.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True))
#
#


###############################################################################
SIMS          = 100
PROCDIST      = "gamma"
PARAMETRIC    = False
NEG_HANDLER   = 1
PERCENTILES   = [.75, .95]
INTERPOLATION = "linear"
PRNG          = RandomState(516)

bcl_init = trikit.chladder(data=DATA, range_method="bootstrap", neg_handler=1)


bcl = bcl_init(
    sims=SIMS, procdist=PROCDIST, parametric=PARAMETRIC,
    percentiles=PERCENTILES, interpolation=INTERPOLATION, random_state=PRNG
    )

bcl.plotdist(level="agg", tc="#FE0000")




# sns.set(rc={'axes.facecolor':"#1f77b4"})
# g = sns.FacetGrid(dat, col="origin", col_wrap=4, margin_titles=False)
# g.map(plt.hist, "reserve", **plt_params)
# #g.fig.subplots_adjust(wspace=.05, hspace=.05)
# g.set_titles("{col_name}", color="red")
# g.fig.suptitle("Reserve Distribution by Origin Year", color="red", weight="bold")
# plt.subplots_adjust(top=0.92)
#
# plt.show()





tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "total_bill", color="steelblue", bins=bins)


# bcl        = trikit.chladder(data=DATA, range_method="bootstrap", neg_handler=1)
# samples    = bcl._bs_samples(sims=SIMS, parametric=PARAMETRIC, random_state=PRNG)
# ldfs       = bcl._bs_ldfs(samples_df=samples)
# clvi       = bcl.tri.clvi.reset_index(drop=False)
# rlvi       = bcl.tri.rlvi.reset_index(drop=False).rename({"index":"origin", "dev":"l_act_dev"},axis=1).drop(labels="col_offset", axis=1)
# combined   = samples.merge(ldfs, how="left", on=["sim", "dev"]).sort_values(by=["sim", "origin", "dev"])
# combined   = combined.merge(rlvi, how="left", on=["origin"]).reset_index(drop=True).sort_values(by=["sim", "origin", "dev"])
# forecasts  = bcl._bs_forecasts(combined_df=combined)
# pedf       = bcl._bs_process_error(forecasts_df=forecasts, procdist="gamma", random_state=PRNG)
# reserves   = bcl._bs_reserves(process_error_df=pedf)






%load_ext line_profiler
%load_ext memory_profiler

# sims = 1000 ==================]
# _bs_samples       : 162-168
# _bs_ldfs          : 136-141
# _bs_forecasts     : 2472-2528
# _bs_process_error : 75-80

# %lprun -s -T E:/Dev/Profiling/l_forecasts1b.txt -f _forecasts1b _forecasts1b(samples=inputdf)
# %lprun -s -T E:/Dev/Profiling/l_forecasts1a.txt -f _forecasts1a _forecasts1a(samples=_samples, ldfs=_ldfs)
#
# %lprun -s -u 1 -f _forecasts1c _forecasts1c(df=combined)
# %lprun -s -u 1 -f _forecasts1c _forecasts1c(df=combined)

tri = bcl_init.tri.copy(deep=True)

ltri = [tri[[i]].copy() for i in tri.columns]
for i in range(len(ltri)):
    iterdf = ltri[i]
    iterdf.columns.name = None
    devp = iterdf.columns[0]
    iterdf.reset_index(drop=False, inplace=True)
    iterdf.rename(columns={"index":"origin", devp:"value"}, inplace=True)
    iterdf["dev"] = devp
    ltri[i] = iterdf
df = pd.concat(ltri, ignore_index=True)
df = df[~np.isnan(df["value"])]
df = df.sort_values(by=["origin", "dev"]).reset_index(drop=True)
dd1 = df[["origin", "dev", "value"]]

del df, ltri, iterdf

tri = bcl_init.tri.copy(deep=True)
tri = tri.reset_index(drop=False).rename({"index":"origin"}, axis=1)
df = pd.melt(tri, id_vars=["origin"], var_name="dev", value_name="value")
df = df[~np.isnan(df["value"])]
dd2 = df.sort_values(by=["origin", "dev"]).reset_index(drop=True)


