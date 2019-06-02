"""
This module contains the class definitions for ``_BaseChainLadder``.
Users should avoid calling any ``*ChainLadder`` instances directly; rather the
dataset and triangle arguments should be passed to ``chladder``, which will
return the initialized ChainLadder instance, from which estimates of outstanding
liabilities and optionally ranges can be obtained.
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


class _BaseChainLadder:
    """
    From the Casualty Actuarial Society's "Estimating Unpaid Claims Using
    Basic Techniques" Version 3 (Friedland, Jacqueline - 2010), the
    development method ('Chain Ladder') consists of seven basic steps:

    1. Compile claims data in a development triangle
    2. Calculate age-to-age factors
    3. Calculate averages of the age-to-age factors
    4. Select claim development factors
    5. Select tail factor
    6. Calculate cumulative claims
    7. Project ultimate claims
    """
    def __init__(self, cumtri):
        """
        Generate point estimates for outstanding claim liabilities at
        ultimate for each origin year and in aggregate. The
        ``_BaseChainLadder`` class exposes no functionality to estimate
        variability around the point estimates at ultimate.

        Parameters
        ----------
        cumtri: triangle._CumTriangle
            A cumulative.CumTriangle instance.
        """
        self.tri = cumtri



    def run(self, sel="all-weighted", tail=1.0):
        """
        Compile a summary of ultimate and reserve estimates resulting from
        the application of the development technique over ``self.tri``.
        Generated DataFrame is comprised of origin year, maturity of origin
        year, loss amount at latest evaluation, cumulative loss development
        factors, projected ultimates and the reserve estimate, by origin
        year and in aggregate.

        Parameters
        ----------
        sel: str
            The ldf average to select from ``triangle._CumTriangle.a2a_avgs``.
            Defaults to "all-weighted".

        tail: float
            Tail factor. Defaults to 1.0.

        Returns
        -------
        trikit.chainladder._ChainLadderResult
        """
        # sel = "all-weighted"
        # tail = 1.0
        # ldfs_ = cl._ldfs(sel=sel, tail=tail)
        # cldfs_ = cl._cldfs(ldfs=ldfs_)
        # ultimates_ = cl._ultimates(cldfs=cldfs_)
        # reserves_ = cl._reserves(ultimates=ultimates_)
        # maturity_ = tri1.maturity.astype(np.str)
        # latest_ = tri1.latest_by_origin

        ldfs_ = self._ldfs(sel=sel, tail=tail)
        cldfs_ = self._cldfs(ldfs=ldfs_)
        ultimates_ = self._ultimates(cldfs=cldfs_)
        reserves_ = self._reserves(ultimates=ultimates_)
        maturity_ = self.tri.maturity.astype(np.str)
        latest_ = self.tri.latest_by_origin
        dfcldfs = cldfs_.to_frame().reset_index(drop=False)
        dfcldfs = dfcldfs.rename({"index":"maturity"}, axis=1)
        dfcldfs["maturity"] = dfcldfs["maturity"].astype(np.str)
        dfmat = maturity_.to_frame().reset_index(drop=False)
        dfmat = dfmat.rename({"index":"origin"}, axis=1)
        dfcldfs = pd.merge(dfmat, dfcldfs, on="maturity", how="inner")
        dfcldfs = dfcldfs.set_index("origin").drop("maturity", axis=1)
        dfcldfs.index.name = None

        # Use join instead of merge to preserve origin year indicies.

        # Compile Chain Ladder point estimate summary.
        # dfmatur_ = maturity_.to_frame().reset_index(drop=False).rename({"index":"origin"}, axis=1)
        # dfcldfs_ = cldfs_.to_frame().reset_index(drop=False).rename({"index":"maturity"}, axis=1)
        # dfcldfs_["maturity"] = dfcldfs_["maturity"].astype(np.str)
        # dfsumm = dfmatur_.merge(dfcldfs_, on=["maturity"], how="left").set_index("origin")
        # dfsumm.index.name = None
        # dflatest_ = latest_.to_frame().rename({"latest_by_origin":"latest"}, axis=1)
        # dfultimates_ = ultimates_.to_frame()
        # dfreserves_ = reserves_.to_frame()
        # dfsumm = functools.reduce(
        #     lambda df1, df2: df1.join(df2),
        #     (dflatest_, dfultimates_, dfreserves_), dfsumm
        #     )
        #

        dfsumm = maturity_.to_frame().join(latest_)
        dfsumm = dfsumm.rename({"latest_by_origin":"latest"}, axis=1)
        dfsumm = dfsumm.join(dfcldfs).join(ultimates_).join(reserves_)
        dfsumm.loc["total"] = dfsumm.sum()

        # Set to NaN fields that shouldn't be aggregated.
        dfsumm.loc["total", "maturity"] = ""
        dfsumm.loc["total", "cldf"] = np.NaN
        dfsumm = dfsumm.reset_index().rename({"index":"origin"}, axis=1)
        kwds = {"sel":sel, "tail":tail}

        # Initialize and return _ChainLadderResult instance.
        clresult_ = _ChainLadderResult(
            summary=dfsumm, tri=self.tri, ldfs=ldfs_, cldfs=cldfs_,
            latest=latest_, maturity=maturity_, ultimates=ultimates_,
            reserves=reserves_, **kwds)
        return(clresult_)



    def _ldfs(self, sel="all-weighted", tail=1.0):
        """
        Lookup loss development factors corresponding to ``sel``.

        Parameters
        ----------
        sel: str
            The ldf average to select from ``triangle._CumTriangle.a2a_avgs``.
            Defaults to "all-weighted".

        tail: float
            Tail factor. Defaults to 1.0.

        Returns
        -------
        pd.Series
        """
        try:
            ldfs_ = self.tri.a2a_avgs.loc[sel]
            tindx_ = ldfs_.index.max() + 1
            ldfs_ = ldfs_.append(pd.Series(data=[tail], index=[tindx_]))
        except KeyError:
                print("Invalid age-to-age selection: `{}`".format(sel))
        ldfs_ =pd.Series(data=ldfs_, index=ldfs_.index, dtype=np.float_, name="ldf")
        return(ldfs_.sort_index())


    def _cldfs(self, ldfs):
        """
        Calculate cumulative loss development factors factors by successive
        multiplication beginning with the tail factor and the oldest
        age-to-age factor. The cumulative claim development factor projects
        the total growth over the remaining valuations. Cumulative claim
        development factors are also known as "Age-to-Ultimate Factors"
        or "Claim Development Factors to Ultimate".

        Parameters
        ----------
        ldfs: pd.Series
            Selected ldfs, typically the output of calling ``self._ldfs``.

        Returns
        -------
        pd.Series
        """
        cldfs_indx = ldfs.index.values
        cldfs_ = np.cumprod(ldfs.values[::-1])[::-1]
        cldfs_ = pd.Series(data=cldfs_, index=ldfs.index.values, name="cldf")
        return(cldfs_.astype(np.float_).sort_index())


    def _ultimates(self, cldfs):
        """
        Ultimate claims are equal to the product of the latest valuation of
        losses (the amount along latest diagonal of any ``_CumTriangle``
        instance) and the appropriate cldf/age-to-ultimate factor. We
        determine the appropriate age-to-ultimate factor based on the age
        of each origin year relative to the evaluation date.

        Parameters
        ----------
        cldfs: pd.Series
            Cumulative loss development factors, conventionally obtained
            via _BaseChainLadder's ``_cldfs`` method.

        Returns
        -------
        pd.Series
        """
        ultimates_ = pd.Series(
            data=self.tri.latest_by_origin.values * cldfs.values[::-1],
            index=self.tri.index, name="ultimate"
            )
        return(ultimates_.astype(np.float_).sort_index())


    def _reserves(self, ultimates):
        """
        Return IBNR/unpaid claim estimates. Represents the difference
        between ultimate loss projections for each origin year and the latest
        cumulative loss amount.
        Note that because outstanding claim liabilities are different based
        on the type of losses represented in the triangle ("ibnr" if losses
        are reported/incurred, "unpaid" if data represents paid losses), we
        use the generic term "reserves" to represent either. In any case,
        what is referred to as "reserves" will always equate to the
        difference between the ultimate loss projections and cumulative
        loss amounts at the latest evaluation period for each origin year.

        Parameters
        ----------
        ultimates: pd.Series
            Estimated ultimate losses, conventionally obtained from
            _BaseChainLadder's ``_ultimates`` method.

        Returns
        -------
        pd.Series
        """
        reserves_ = pd.Series(
            data=ultimates - self.tri.latest_by_origin,
            index=self.tri.index, name='reserve')
        return(reserves_.astype(np.float_).sort_index())










class _ChainLadderResult:
    """
    Curated output resulting from ``_BaseChainLadder``'s ``run`` method.
    """
    def __init__(self, summary, tri, ldfs, cldfs, latest, maturity,
                 ultimates, reserves, **kwargs):
        """
        Container object for ``_BaseChainLadder`` output.

        Parameters
        ----------
        summary: pd.DataFrame
            Chain Ladder summary compilation.

        tri: trikit.triangle._CumTriangle
            A cumulative triangle instance.

        ldfs: pd.Series
            Loss development factors.

        cldfs: pd.Series
            Cumulative loss development factors.

        latest: pd.Series
            Latest loss amounts by origin.

        maturity: pd.Series
            Represents ther maturity of each origin relative to development
            period.

        ultimates: pd.Series
            Represents Chain Ladder ultimate projections.

        reserves: pd.Series
            Represents the projected reserve amount. For each origin period,
            this equates to the ultimate loss projection minus the latest
            loss amount for the origin period (reserve = ultimate - latest).

        kwargs: dict
            Additional keyword arguments passed into ``_BaseChainLadder``'s
            ``run`` method.
        """
        self.ultimates = ultimates
        self.reserves = reserves
        self.summary = summary
        self.cldfs = cldfs
        self.ldfs = ldfs
        self.tri = tri

        if kwargs is not None:
            for key_ in kwargs:
                setattr(self, key_, kwargs[key_])

        # Properties.
        self._trisqrd = None


    @property
    def trisqrd(self):
        """
        Project claims growth for each future development period. Returns a
        DataFrame of loss projections for each subsequent development period
        for each accident year. Populates the triangle's lower-right or
        southeast portion (i.e., "squaring the triangle").

        Returns
        -------
        pd.DataFrame
        """
        if self._trisqrd is None:
            self._trisqrd = self.tri.copy(deep=True)
            ldfs = self.ldfs.values
            rposf = self.tri.index.size
            clvi = self.tri.clvi["row_offset"]
            for i in enumerate(self._trisqrd.columns[1:], start=1):
                ii  , devp  = i[0], i[1]
                ildf, rposi = ldfs[ii - 1], clvi[devp] + 1
                self._trisqrd.iloc[rposi:rposf, ii] = \
                    self._trisqrd.iloc[rposi:rposf, ii - 1] * ildf
            # Multiply right-most column by tail factor.
            max_devp = self._trisqrd.columns[-1]
            self._trisqrd["ultimate"] = self._trisqrd.loc[:,max_devp].values * self.tail
        return(self._trisqrd.astype(np.float_).sort_index())



    def plot(self, actuals_color="#334488", forecasts_color="#FFFFFF",
             axes_style="darkgrid", context="notebook", col_wrap=5,
             **kwargs):
        """
        Visualize projected chain ladder development. First transforms data
        into long format, then plots actual and forecast loss amounts at each
        development period for each origin year using seaborn's ``FacetGrid``.

        Parameters
        ----------
        actuals_color: str
            A hexidecimal color code used to represent actual scatter points
            in FacetGrid. Defaults to "#00264C".

        forecasts_color: str
            A hexidecimal color code used to represent forecast scatter points
            in FacetGrid. Defaults to "#FFFFFF".

        axes_style: str
            Aesthetic style of plots. Defaults to "darkgrid". Other options
            include: {whitegrid, dark, white, ticks}.

        context: str
            Set the plotting context parameters. According to the seaborn
            documentation, This affects things like the size of the labels,
            lines, and other elements of the plot, but not the overall style.
            Defaults to ``"notebook"``. Additional options include
            {paper, talk, poster}.

        col_wrap: int
            The maximum number of origin period axes to have on a single row
            of the resulting FacetGrid. Defaults to 5.

        kwargs: dict
            Additional styling options for scatter points. This can override
            default values for ``plt.plot`` objects. For a demonstration,
            See the Examples section.

        Returns
        -------
        None

        Examples
        --------
        Demonstration of how to pass a dictionary of plot properties in order
        to update the scatter size and marker:

        >>> import trikit
        >>> raa = trikit.load(dataset="raa")
        >>> cl_init = trikit.chladder(data=raa)
        >>> cl_result = cl_init(sel="simple-5", tail=1.005)

        ``cl_result`` represents an instance of ``_ChainLadderResult``, which
        exposes the ``plot`` method. First, we compile the dictionary of
        attributes to override:

        >>> kwds = dict(marker="s", markersize=6)
        >>> cl_result.plot(**kwds)
        """
        df0 = self.trisqrd.reset_index(drop=False).rename({"index":self.tri.origin}, axis=1)
        df0 = pd.melt(df0, id_vars=[self.tri.origin], var_name=self.tri.dev, value_name=self.tri.value)
        df0 = df0[~np.isnan(df0[self.tri.value])].reset_index(drop=True)
        df1 = self.tri.triind.reset_index(drop=False).rename({"index":self.tri.origin}, axis=1)
        df1 = pd.melt(df1, id_vars=[self.tri.origin], var_name=self.tri.dev, value_name=self.tri.value)
        df1[self.tri.value] = df1[self.tri.value].map(lambda v: 1 if v==0 else 0)
        df1 = df1[~np.isnan(df1[self.tri.value])].rename({self.tri.value:"actual_ind"}, axis=1)
        df1 = df1.reset_index(drop=True)
        if self.tail!=1:
            df0[self.tri.dev] = df0[self.tri.dev].map(
                lambda v: (self.tri.devp.max() + 1) if v=="ultimate" else v
                )
        else:
            df0 = df0[df0[self.tri.dev]!="ultimate"]

        # Combine df0 and df1 into a single DataFrame, then perform cleanup
        # actions for cases in which df0 has more records than df1.
        df = pd.merge(df0, df1, on=[self.tri.origin, self.tri.dev], how="left", sort=False)
        df["actual_ind"] = df["actual_ind"].map(lambda v: 0 if np.isnan(v) else v)
        df["actual_ind"] = df["actual_ind"].astype(np.int_)
        df = df.sort_values([self.tri.origin, self.tri.dev]).reset_index(drop=True)
        dfma = df[df["actual_ind"]==1].groupby([self.tri.origin])[self.tri.dev].max().to_frame()
        dfma = dfma.reset_index(drop=False).rename(
            {"index":self.tri.origin, self.tri.dev:"max_actual"}, axis=1)
        df = pd.merge(df, dfma, on=self.tri.origin, how="outer", sort=False)
        df = df.sort_values([self.tri.origin, self.tri.dev]).reset_index(drop=True)
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


    def __str__(self):
        formats_ = {"ultimate":"{:.0f}".format, "reserve":"{:.0f}".format,
                    "latest":"{:.0f}".format, "cldf":"{:.5f}".format,}
        return(self.summary.to_string(formatters=formats_))


    def __repr__(self):
        formats_ = {"ultimate":"{:.0f}".format, "reserve":"{:.0f}".format,
                    "latest":"{:.0f}".format, "cldf":"{:.5f}".format,}
        return(self.summary.to_string(formatters=formats_))
