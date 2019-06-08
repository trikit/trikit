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
import matplotlib.pyplot as plt
import seaborn as sns
import trikit



"""
 ### TODO ###
- Check that cl.ultimates == trisqrd[-1]
- Check that cl.trisqrd - cl.tri == 0
- Check that cl0.ultimates-cl0.reserves-cl0.latest == 0
"""

RAA = trikit.load("raa")
ta83 = trikit.load("ta83")
tri = trikit.totri(RAA)


bcl = tri.chladder(range_method="bootstrap", sims=1000)
dfsim = bcl.sims_data[["origin", "dev", "reserve"]]
dfres = bcl.reserve_dist


# dfsim["u95"] = dfsim.groupby(["origin", "dev"], as_index=False)["reserve"].transform("quantile", pctlu)
dfsim = bcl.sims_data[["origin", "dev", "reserve"]]







def get_quantile(q, symmetric=True, interpolation="linear"):
    """
    Return percentile of bootstrapped reserve distribution given by
    pctl.

    Parameters
    ----------
    q: float in range of [0,1] (or sequence of floats)
        Percentile to compute, which must be between 0 and 1 inclusive.

    symmetric: bool
        Whether the symmetric interval should be returned. For example, if
        ``symmetric==True`` and ``q=.95``, then the 2.5th and 97.5th
        quantiles of the bootstrapped reserve distribution will be returned
        [(1 - .95) / 2, (1 + .95) / 2]. When False, only the specified
        quantile(s) will be computed.

    interpolation: {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to use when
        the desired quantile lies between two data points i < j:

            - linear: i + (j - i) * fraction, where fraction is the fractional
            part of the index surrounded by i and j.
            - lower: i.
            - higher: j.
            - nearest: i or j, whichever is nearest.
            - midpoint: (i + j) / 2.

    Returns
    -------
    pd.DataFrame
    """
    # dfsims = bcl.sims_data[["origin", "dev", "reserve"]]
    # symmetric =True
    # q = [.90, .95]
    # interpolation = "linear"

    dfsims = self.sims_data[["origin", "dev", "reserve"]]
    pctl_ = np.asarray([q] if isinstance(q, (float, int)) else q)
    if np.any(np.logical_or(pctl_ > 1, pctl_ < 0)):
        raise ValueError("Values for percentiles must fall between [0, 1].")
    else:
        if symmetric:
            pctlarr = np.sort(np.unique(np.append((1 - pctl_) / 2, (1 + pctl_) / 2)))
        else:
            pctlarr = np.sort(np.unique(pctl_))

        pctlfmt = [
            "{:.5f}".format(i).rstrip("0").rstrip(".") + "%" for i in 100 * pctlarr
            ]

        # Initialize DataFrame for percentile data.
        dfpctl = dfsims.groupby(["origin", "dev"]).aggregate(
            "quantile", q=.50, interpolation=interpolation)
        dfpctl = dfpctl.rename({"reserve":"50%"}, axis=1)
        dfpctl.columns.name = None

        for q_, pctlstr_ in zip(pctlarr, pctlfmt):
            if q_!=.50:
                df_ = dfsims.groupby(["origin", "dev"]).aggregate(
                    "quantile", q=q_, interpolation=interpolation)
                df_ = df_.rename({"reserve":pctlstr_}, axis=1)
                df_.columns.name = None
                dfpctl = dfpctl.join(df_)

        if .50 not in pctl_:
            dfpctl = dfpctl.drop("50%", axis=1)

        return(dfpctl.reset_index(drop=False).sort_index())

















    ll = [dfsim.groupby(["origin", "dev"]).aggregate("quantile", q=i)

    for pctl_, pctlstr_ in zip(pctlarr, pctlfmt):
        dfsumm[pctlstr_] = dfsumm.index.map(
            lambda v: np.percentile(
                dfreserves[dfreserves["origin"]==v]["reserve"].values,
                pctl_, interpolation=interpolation
                )
            )

        










df0 = bcl.trisqrd.reset_index(drop=False).rename({"index":bcl.tri.origin}, axis=1)
df0 = pd.melt(df0, id_vars=[bcl.tri.origin], var_name=bcl.tri.dev, value_name=bcl.tri.value)
df0 = df0[~np.isnan(df0[bcl.tri.value])].reset_index(drop=True)
df1 = bcl.tri.triind.reset_index(drop=False).rename({"index":bcl.tri.origin}, axis=1)
df1 = pd.melt(df1, id_vars=[bcl.tri.origin], var_name=bcl.tri.dev, value_name=bcl.tri.value)
df1[bcl.tri.value] = df1[bcl.tri.value].map(lambda v: 1 if v==0 else 0)
df1 = df1[~np.isnan(df1[bcl.tri.value])].rename({bcl.tri.value:"actual_ind"}, axis=1)
df1 = df1.reset_index(drop=True)
if bcl.tail!=1:
    df0[bcl.tri.dev] = df0[bcl.tri.dev].map(
        lambda v: (bcl.tri.devp.max() + 1) if v=="ultimate" else v
        )
else:
    df0 = df0[df0[bcl.tri.dev]!="ultimate"]

# Combine df0 and df1 into a single DataFrame, then perform cleanup
# actions for cases in which df0 has more records than df1.
df = pd.merge(df0, df1, on=[bcl.tri.origin, bcl.tri.dev], how="left", sort=False)
df["actual_ind"] = df["actual_ind"].map(lambda v: 0 if np.isnan(v) else v)
df["actual_ind"] = df["actual_ind"].astype(np.int_)
df = df.sort_values([bcl.tri.origin, bcl.tri.dev]).reset_index(drop=True)
dfma = df[df["actual_ind"]==1].groupby([bcl.tri.origin])[bcl.tri.dev].max().to_frame()
dfma = dfma.reset_index(drop=False).rename(
    {"index":bcl.tri.origin, bcl.tri.dev:"max_actual"}, axis=1)
df = pd.merge(df, dfma, on=bcl.tri.origin, how="outer", sort=False)
df = df.sort_values([bcl.tri.origin, bcl.tri.dev]).reset_index(drop=True)
df["incl_actual"] = df["actual_ind"].map(lambda v: 1 if v==1 else 0)
df["incl_pred"] = df.apply(
    lambda rec: 1 if (rec.actual_ind==0 or rec.dev==rec.max_actual) else 0,
    axis=1
    )

# Vertically concatenate dfact_ and dfpred_.
dfact_ = df[df["incl_actual"]==1][["origin", "dev", "value"]]
dfact_["description"] = "actual"
dfpred_ = df[df["incl_pred"]==1][["origin", "dev", "value"]]
dfpred_["description"] = "forecast"
data = pd.concat([dfact_, dfpred_]).reset_index(drop=True)

# Plot chain ladder projections by development period for each
# origin year. FacetGrid's ``hue`` argument should be set to
# "description".
axes_style="darkgrid"
context="notebook"

sns.set_context(context)

with sns.axes_style(axes_style):
    titlestr_ = "Chain Ladder Projections with Actuals by Origin"
    palette_ = dict(actual=actuals_color, forecast=forecasts_color)
    pltkwargs = dict(
        marker="o", markersize=7, alpha=1, markeredgecolor="#000000",
        markeredgewidth=.50, linestyle="--", linewidth=.75,
        fillstyle="full",
        )

    if kwargs:
        pltkwargs.update(kwargs)

    g = sns.FacetGrid(
        data, col="origin", hue="description", palette=palette_,
        col_wrap=col_wrap, margin_titles=False, despine=True, sharex=True,
        sharey=True, hue_order=["forecast", "actual",]
        )

    g.map(plt.plot, "dev", "value", **pltkwargs)
    g.set_axis_labels("", "")
    g.set(xticks=data.dev.unique().tolist())
    g.set_titles("{col_name}", size=9)
    g.set_xticklabels(data.dev.unique().tolist(), size=8)

    # Change ticklabel font size and place legend on each facet.
    for i, _ in enumerate(g.axes):
        ax_ = g.axes[i]
        legend_ = ax_.legend(
            loc="lower right", fontsize="x-small", frameon=True,
            fancybox=True, shadow=False, edgecolor="#909090",
            framealpha=1, markerfirst=True,)
        legend_.get_frame().set_facecolor("#FFFFFF")
        ylabelss_ = [i.get_text() for i in list(ax_.get_yticklabels())]
        ylabelsn_ = [float(i.replace(u"\u2212", "-")) for i in ylabelss_]
        ylabelsn_ = [i for i in ylabelsn_ if i>=0]
        ylabels_ = ["{:,.0f}".format(i) for i in ylabelsn_]
        ax_.set_yticklabels(ylabels_, size=8)

        # Draw border around each facet.
        for _, spine_ in ax_.spines.items():
            spine_.set_visible(True)
            spine_.set_color("#000000")
            spine_.set_linewidth(.50)

# Adjust facets downward and and left-align figure title.
plt.subplots_adjust(top=0.9)
g.fig.suptitle(
    titlestr_, x=0.065, y=.975, fontsize=11, color="#404040", ha="left"
    )



# Instantiate MackChainLadder instance =======================================]
tri = trikit.totri(ta83)
mcl = trikit._MackChainLadder(cumtri=tri)

ldfs0 = mcl._ldfs(alpha=0, tail=1.0)
ldfs1 = mcl._ldfs(alpha=1, tail=1.0)
ldfs2 = mcl._ldfs(alpha=2, tail=1.0)


dpv0 = mcl._devpvar(alpha=0, tail=1.0)
dpv1 = mcl._devpvar(alpha=1, tail=1.0)
dpv2 = mcl._devpvar(alpha=2, tail=1.0)





cldfs   = mcl0.cldfs
invsums = mcl0.inverse_sums
devpvar  = mcl0.devpvar
procerr = mcl0.process_error
originref = mcl0.originref
devpref = mcl0.devpref
rmse    = mcl0.rmsepi

def f(*args):
    return([i for i in args])
















#
class ChainLadderTestCase(unittest.TestCase):
    def setUp(self):
        self.cl = trikit.BaseChainLadder(data=raa, sel="all-weighted", tail=1.0)


    def test_trisqrd(self):
        """
        Verify that self.tri and self.trisqrd are the same in
        upper left.
        """
        self.assertEqual(
            (self.cl.trisqrd-self.cl.tri).sum().sum(),0.0,
            "trisqrd not correctly implemented."
            )


    def test_ultimates(self):
        """
        Verify that ultimates matches the last column of trisqrd.
        """
        atults = self.cl.ultimates
        tsults = self.cl.trisqrd.loc[:,self.cl.trisqrd.columns[-1]]
        self.assertEqual(
            atults, tsults, "Difference in ultimate results"
            )

    def test_reserves(self):
        """
        Test value consistency.
        """
        assertEqual(
            (self.cl.ultimates-self.cl.latest_by_origin-self.cl.reserves).sum(),
            0, "Inconsistency in ults, latest and reserves."
            )






# Bootstrap Unit Tests ########################################################

class BootChainLadderTestCase(unittest.TestCase):
    def setUp(self):
        self.bcl = trikit.BootChainLadder(data=raa, sel="all-weighted", tail=1.0)


    def test_trisqrd(self):
        """
        Verify that self.tri and self.trisqrd are the same in
        upper left.
        """
        self.assertEqual(
            (self.cl.trisqrd-self.cl.tri).sum().sum(),0.0,
            "trisqrd not correctly implemented."
            )


    def test_ultimates(self):
        """
        Verify that ultimates matches the last column of trisqrd.
        """
        atults = self.cl.ultimates
        tsults = self.cl.trisqrd.loc[:,self.cl.trisqrd.columns[-1]]
        self.assertEqual(
            atults, tsults, "Difference in ultimate results"
            )

    def test_reserves(self):
        """
        Test value consistency.
        """
        assertEqual(
            (self.cl.ultimates-self.cl.latest_by_origin-self.cl.reserves).sum(),
            0, "Inconsistency in ults, latest and reserves."
            )
