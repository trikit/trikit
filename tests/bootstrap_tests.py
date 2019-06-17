import sys; sys.path.append("G:\\Repos")   #"C:\\Users\\cac9159\\Repos\\")
import unittest
import trikit
import pandas as pd
import numpy as np
import os
import os.path
import decimal
import logging
import timeit
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import trikit


RAA = trikit.load("raa")
ta83 = trikit.load("ta83")
tri = trikit.totri(RAA)
# ============================================================================]


tri = trikit.totri(RAA)

cl = tri.chladder(sel="all-weighted", tail=1.0)
bcl = tri.chladder(range_method="bootstrap", sims=1000)









# ============================================================================]
# bcl = tri.chladder(range_method="bootstrap", q=[.75, .95], symmetric=False, sims=1000)
# tfc = bcl._tri_fit_cum(sel="all-weighted")
# tfi = bcl._tri_fit_incr(tfc)
# r_us = bcl._resid_us(tfi)
# sclp = bcl._scale_param(r_us)
# r_adj = bcl._resid_adj(r_us)
# sampd = bcl._sampling_dist(r_adj)
# dfsamples = bcl._bs_samples(sampd, tfi, sims=100)
# dfldfs = bcl._bs_ldfs(dfsamples)
# dflvi = bcl.tri.rlvi.reset_index(drop=False)
# dflvi = dflvi.rename({"index":"origin", "dev":"l_act_dev"}, axis=1)
# dflvi = dflvi.drop("col_offset", axis=1)
# dfcombined = dfsamples.merge(dfldfs, on=["sim", "dev"], how="left")
# dfcombined = dfcombined.merge(dflvi, on=["origin"], how="left", )
# dfcombined = dfcombined.reset_index(drop=True).sort_values(by=["sim", "origin", "dev"])
# dfforecasts = bcl._bs_forecasts(dfcombined=dfcombined, scale_param=sclp)
# dfprocerror = bcl._bs_process_error(dfforecasts=dfforecasts, scale_param=sclp,)
# ============================================================================]





# df0 = bcl.trisqrd.reset_index(drop=False).rename({"index":bcl.tri.origin}, axis=1)
# df0 = pd.melt(df0, id_vars=[bcl.tri.origin], var_name=bcl.tri.dev, value_name=bcl.tri.value)
# df0 = df0[~np.isnan(df0[bcl.tri.value])].reset_index(drop=True)
# df1 = bcl.tri.triind.reset_index(drop=False).rename({"index":bcl.tri.origin}, axis=1)
# df1 = pd.melt(df1, id_vars=[bcl.tri.origin], var_name=bcl.tri.dev, value_name=bcl.tri.value)
# df1[bcl.tri.value] = df1[bcl.tri.value].map(lambda v: 1 if v==0 else 0)
# df1 = df1[~np.isnan(df1[bcl.tri.value])].rename({bcl.tri.value:"actual_ind"}, axis=1)
# df1 = df1.reset_index(drop=True)
# if bcl.tail!=1:
#     df0[bcl.tri.dev] = df0[bcl.tri.dev].map(
#         lambda v: (bcl.tri.devp.max() + 1) if v=="ultimate" else v
#         )
# else:
#     df0 = df0[df0[bcl.tri.dev]!="ultimate"]
#
# # Combine df0 and df1 into a single DataFrame, then perform cleanup
# # actions for cases in which df0 has more records than df1.
# # df = pd.merge(df0, df1, on=[bcl.tri.origin, bcl.tri.dev], how="left", sort=False)
# # df["actual_ind"] = df["actual_ind"].map(lambda v: 0 if np.isnan(v) else v)
# # df["actual_ind"] = df["actual_ind"].astype(np.int_)
# # df = df.sort_values([bcl.tri.origin, bcl.tri.dev]).reset_index(drop=True)
# # dfma = df[df["actual_ind"]==1].groupby([bcl.tri.origin])[bcl.tri.dev].max().to_frame()
# # dfma = dfma.reset_index(drop=False).rename(
# #     {"index":bcl.tri.origin, bcl.tri.dev:"max_actual"}, axis=1)
# # df = pd.merge(df, dfma, on=bcl.tri.origin, how="outer", sort=False)
# # df = df.sort_values([bcl.tri.origin, bcl.tri.dev]).reset_index(drop=True)
# # df["incl_actual"] = df["actual_ind"].map(lambda v: 1 if v==1 else 0)
# # df["incl_pred"] = df.apply(
# #     lambda rec: 1 if (rec.actual_ind==0 or rec.dev==rec.max_actual) else 0,
# #     axis=1
# #     )
# #
# # # Vertically concatenate dfact_ and dfpred_.
# # dfact_ = df[df["incl_actual"]==1][["origin", "dev", "value"]]
# # dfact_["rectype"] = "actual"
# # dfpred_ = df[df["incl_pred"]==1][["origin", "dev", "value"]]
# # dfpred_["rectype"] = "forecast"
# # data = pd.concat([dfact_, dfpred_]).reset_index(drop=True)
#
#
# data = pd.merge(data, dfsims, how="outer", on=["origin", "dev"])
# pctl_hdrs = [i for i in dfsims.columns if i not in ("origin", "dev")]
# for hdr_ in pctl_hdrs:
#     data[hdr_] = np.where(
#         data["rectype"].values=="actual", np.NaN, data[hdr_].values
#         )
#
# # Determine the first forecast period by origin, and set q-fields to actuals.
# data["_ff"] = np.where(data["rectype"].values=="forecast", data["dev"].values, data.dev.values.max() + 1)
# data["_minf"] = data.groupby(["origin"])["_ff"].transform("min")
# for hdr_ in pctl_hdrs:
#     data[hdr_] = np.where(
#         np.logical_and(data["rectype"].values=="forecast", data["_minf"].values==data["dev"].values),
#         data["value"].values, data[hdr_].values
#         )
#
#
# data.drop(["_ff", "_minf"], axis=1, inplace=True)
# dfv = data[["origin", "dev", "rectype", "value"]]
# dfl = data[["origin", "dev", "rectype", pctl_hdrs[0]]]
# dfu = data[["origin", "dev", "rectype", pctl_hdrs[-1]]]
# dfl["rectype"] = pctl_hdrs[0]
# dfl = dfl.rename({pctl_hdrs[0]:"value"}, axis=1)
# dfu["rectype"] = pctl_hdrs[-1]
# dfu = dfu.rename({pctl_hdrs[-1]:"value"}, axis=1)
# dfstk = pd.concat([dfv, dfl, dfu])




# Plot chain ladder projections by development period for each
# origin year. FacetGrid's ``hue`` argument should be set to
# "description".
axes_style="darkgrid"
context="notebook"
actuals_color="#334488"
forecasts_color="#FFFFFF"
fill_color = "#B1FCB1"
col_wrap=5
fillcolors = {"green":"#B1FCB1", "peach":"#FCD7B1", "purple":"#D7B1FC", "yellow":"#FCFCB1",
              "orange":"#FFD282", "pink":"#FFC1CB",}
fillcolor = fillcolors["pink"]
kwargs=None

sns.set_context(context)
sns.axes_style(axes_style)

#
# with sns.axes_style(axes_style):
#
#     titlestr_ = "Bootstrap Chain Ladder Ultimate Range Projections by Origin"
#     # palette_ = {"actual":actuals_color, "forecast":forecasts_color,
#     #             pctl_hdrs[0]:"#000000", pctl_hdrs[-1]:"#000000"}
#     palette_ = None
#     # pltkwargs = dict(
#     #     marker="o", markersize=6, alpha=1, markeredgecolor="#000000",
#     #     markeredgewidth=.50, linestyle="--", linewidth=.65,
#     #     fillstyle="full",
#     #     )
#     # ulbkwargs = dict(
#     #     alpha=1, color="#000000", linestyle="-", linewidth=.65,
#     #     )
#     fillkwargs = dict(
#         alpha=.75, color="#FCFCB1", label="nolegend",
#         )
#
#     huekwargs = dict(
#         marker=["o", "o", None, None,], markersize=[6, 6, None, None,],
#         color=["#000000", "#000000", "#000000", "#000000",],
#         fillstyle=["full", "full", "none", "none",],
#         markerfacecolor=[forecasts_color, actuals_color, None, None,],
#         markeredgecolor=["#000000", "#000000", None, None,],
#         markeredgewidth=[.50, .50, None, None,],
#         linestyle=["-", "-", "-.", "--",], linewidth=[.475, .475, .625, .625,],
#         )
#
#
#     # if kwargs:
#     #     pltkwargs.update(kwargs)
#
#
#     g = sns.FacetGrid(
#         dfstk, col="origin", palette=palette_, hue="rectype", hue_kws=huekwargs,
#         col_wrap=col_wrap, margin_titles=False, despine=True,
#         sharex=True, sharey=True,
#         hue_order=["forecast", "actual", pctl_hdrs[0], pctl_hdrs[-1]]
#         )
#
#     mean_ = g.map(plt.plot, "dev", "value",)
#     # lbound_= g.map(plt.plot, "dev", dfstk.columns[-2], label="_nolegend_", **ulbkwargs)
#     # ubound_ = g.map(plt.plot, "dev", dfstk.columns[-1], label="_nolegend_", **ulbkwargs)
#
#     # fill_between(x, y1, y2=0, where=None, interpolate=False, step=None, *, dfstk=None,
#     g.set_axis_labels("", "")
#     g.set(xticks=dfstk.dev.unique().tolist())
#     g.set_titles("{col_name}", size=9)
#     g.set_xticklabels(dfstk.dev.unique().tolist(), size=8)
#
#     # Change ticklabel font size and place legend on each facet.
#     for i, _ in enumerate(g.axes):
#         ax_ = g.axes[i]
#         legend_ = ax_.legend(
#             loc="upper left", fontsize="x-small", frameon=True,
#             fancybox=True, shadow=False, edgecolor="#909090",
#             framealpha=1, markerfirst=True,)
#         legend_.get_frame().set_facecolor("#FFFFFF")
#
#         ylabelss_ = [i.get_text() for i in list(ax_.get_yticklabels())]
#         ylabelsn_ = [float(i.replace(u"\u2212", "-")) for i in ylabelss_]
#         ylabelsn_ = [i for i in ylabelsn_ if i>=0]
#         ylabels_ = ["{:,.0f}".format(i) for i in ylabelsn_]
#         ax_.set_yticklabels(ylabels_, size=8)
#
#         # Fill between upper and lower range bounds.
#         c = ax_.get_children()
#         lines_ = [i for i in c if isinstance(i, matplotlib.lines.Line2D)]
#         xx = [i._x for i in lines_ if len(i._x)>0]
#         yy = [i._y for i in lines_ if len(i._y)>0]
#         x_, lb_, ub_ = xx[0], yy[-2], yy[-1]
#         ax_.fill_between(x_, lb_, ub_, **fillkwargs)
#
#         # Draw border around each facet.
#         for _, spine_ in ax_.spines.items():
#             spine_.set_visible(True)
#             spine_.set_color("#000000")
#             spine_.set_linewidth(.50)
#
#
#         # handles, labels = ax_.get_legend_handles_labels()
#         # for h in handles: h.set_linestyle("")
#         #
#         # ll = ax_.get_legend()
#         # c = ll.get_children()
#
#
# # Adjust facets downward and and left-align figure title.
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle(
#     titlestr_, x=0.065, y=.975, fontsize=10, color="#404040", ha="left"
#     )


ll = list(enumerate(g.axes))
i, _ = ll[8]
ax = g.axes[i]
c = ax.get_children()

lines = [i for i in c if isinstance(i, matplotlib.lines.Line2D)]
xx = [i._x for i in lines if len(i._x)>0]
yy = [i._y for i in lines if len(i._y)>0]

x_, lb_, ub_ = xx[0], yy[-2], yy[-1]
l1, l2 = lines[0], lines[1]