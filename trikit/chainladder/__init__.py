"""
This module contains the class definitions for ``BaseChainLadder``.-
"""
import functools
import pandas as pd
import numpy as np



class BaseChainLadder:
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
        ``BaseChainLadder`` class exposes no functionality to estimate
        variability around the point estimates at ultimate.

        Parameters
        ----------
        cumtri: triangle._CumTriangle
            A cumulative.CumTriangle instance.
        """
        self.tri = cumtri



    def __call__(self, sel="all-weighted", tail=1.0):
        """
        Compile a summary of ultimate and reserve estimates resulting from
        the application of the development technique over a triangle instance.
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
        trikit.chainladder.BaseChainLadderResult
        """
        ldfs_ = self._ldfs(sel=sel, tail=tail)
        cldfs_ = self._cldfs(ldfs=ldfs_)
        ultimates_ = self._ultimates(cldfs=cldfs_)
        reserves_ = self._reserves(ultimates=ultimates_)
        maturity_ = self.tri.maturity.astype(np.str)
        latest_ = self.tri.latest_by_origin
        trisqrd_ = self._trisqrd(ldfs=ldfs_)

        # Compile chain ladder point estimate summary.
        dfmatur_ = maturity_.to_frame().reset_index(drop=False).rename({"index":"origin"}, axis=1)
        dfcldfs_ = cldfs_.to_frame().reset_index(drop=False).rename({"index":"maturity"}, axis=1)
        dfcldfs_["maturity"] = dfcldfs_["maturity"].astype(np.str)
        dfsumm = dfmatur_.merge(dfcldfs_, on=["maturity"], how="left").set_index("origin")
        dfsumm.index.name = None
        dflatest_ = latest_.to_frame().rename({"latest_by_origin":"latest"}, axis=1)
        dfultimates_, dfreserves_ = ultimates_.to_frame(), reserves_.to_frame()
        dfsumm = functools.reduce(
            lambda df1, df2: df1.join(df2),
            (dflatest_, dfultimates_, dfreserves_), dfsumm
            )

        dfsumm.loc["total"] = dfsumm.sum()
        dfsumm.loc["total", "maturity"] = ""
        dfsumm.loc["total", "cldf"] = np.NaN
        dfsumm = dfsumm.reset_index().rename({"index":"origin"}, axis=1)
        kwds = {"sel":sel, "tail":tail}

        # Initialize and return _ChainLadderResult instance.
        clresult_ = BaseChainLadderResult(
            summary=dfsumm, tri=self.tri, ldfs=ldfs_, cldfs=cldfs_,
            latest=latest_, maturity=maturity_, ultimates=ultimates_,
            reserves=reserves_, trisqrd=trisqrd_, **kwds)
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
        ldfs_ = pd.Series(data=ldfs_, index=ldfs_.index, dtype=np.float_, name="ldf")
        return(ldfs_.sort_index())


    def _cldfs(self, ldfs):
        """
        Calculate cumulative loss development factors by successive
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
            via BaseChainLadder's ``_cldfs`` method.

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
        Return IBNR/reserve estimates by origin and in aggregate. Represents
        the difference between ultimate projections for each origin period
        and the latest cumulative value.
        Since outstanding claim liabilities can be referred to differently
        based on the type of losses represented in the triangle ("ibnr" if
        reported/incurred, "unpaid" if paid losses), we use the general term
        "reserve" to represent the difference between ultimate projections
        and latest cumulative value by origin and in total.

        Parameters
        ----------
        ultimates: pd.Series
            Estimated ultimate losses, conventionally obtained from
            BaseChainLadder's ``_ultimates`` method.

        Returns
        -------
        pd.Series
        """
        reserves_ = pd.Series(
            data=ultimates - self.tri.latest_by_origin,
            index=self.tri.index, name='reserve')
        return(reserves_.astype(np.float_).sort_index())


    def _trisqrd(self, ldfs):
        """
        Project claims growth for each future development period. Returns a
        DataFrame of loss projections for each subsequent development period
        for each accident year. Populates the triangle's lower-right or
        southeast portion (i.e., the result of "squaring the triangle").

        Returns
        -------
        pd.DataFrame
        """
        trisqrd_ = self.tri.copy(deep=True)
        rposf = self.tri.index.size
        clvi = self.tri.clvi["row_offset"]
        for i in enumerate(trisqrd_.columns[1:], start=1):
            ii  , devp  = i[0], i[1]
            ildf, rposi = ldfs.values[ii - 1], clvi[devp] + 1
            trisqrd_.iloc[rposi:rposf, ii] = \
                trisqrd_.iloc[rposi:rposf, ii - 1] * ildf
        # Multiply right-most column by tail factor.
        max_devp = trisqrd_.columns[-1]
        trisqrd_["ultimate"] = trisqrd_.loc[:,max_devp].values * ldfs.values[-1]
        return(trisqrd_.astype(np.float_).sort_index())



class BaseChainLadderResult:
    """
    Summary class consisting of output resulting from invocation of
    ``BaseChainLadder``'s ``__call__`` method.
    """
    def __init__(self, summary, tri, ldfs, cldfs, latest, maturity,
                 ultimates, reserves, trisqrd, **kwargs):
        """
        Container object for ``BaseChainLadder`` output.

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
            Additional keyword arguments passed into ``BaseChainLadder``'s
            ``run`` method.
        """
        self.ultimates = ultimates
        self.reserves = reserves
        self.summary = summary
        self.trisqrd = trisqrd
        self.cldfs = cldfs
        self.ldfs = ldfs
        self.tri = tri

        if kwargs is not None:
            for key_ in kwargs:
                setattr(self, key_, kwargs[key_])

        self._summspecs = {
            "ultimate":"{:.0f}".format, "reserve":"{:.0f}".format,
            "latest":"{:.0f}".format, "cldf":"{:.5f}".format,
            }


    def _data_transform(self):
        """
        Transform dataset for use in FacetGrid plot by origin exhibting chain
        ladder ultimate & reserve estimates.

        Returns
        -------
        pd.DataFrame
        """
        df0 = self.trisqrd.reset_index(drop=False).rename({"index":"origin" }, axis=1)
        df0 = pd.melt(df0, id_vars=["origin"], var_name="dev", value_name="value")
        df0 = df0[~np.isnan(df0["value"])].reset_index(drop=True)
        df1 = self.tri.triind.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        df1 = pd.melt(df1, id_vars=["origin"], var_name="dev", value_name="value")
        df1["value"] = df1["value"].map(lambda v: 1 if v==0 else 0)
        df1 = df1[~np.isnan(df1["value"])].rename({"value":"actual_ind"}, axis=1)
        df1 = df1.reset_index(drop=True)
        if self.tail!=1:
            df0["dev"] = df0["dev"].map(
                lambda v: (self.tri.devp.max() + 1) if v=="ultimate" else v
                )
        else:
            df0 = df0[df0["dev"]!="ultimate"]

        # Combine df0 and df1 into a single DataFrame, then perform cleanup
        # actions for cases in which df0 has more records than df1.
        df = pd.merge(df0, df1, on=["origin", "dev"], how="left", sort=False)
        df["actual_ind"] = df["actual_ind"].map(lambda v: 0 if np.isnan(v) else v)
        df["actual_ind"] = df["actual_ind"].astype(np.int_)
        df = df.sort_values(["origin", "dev"]).reset_index(drop=True)
        dfma = df[df["actual_ind"]==1].groupby(["origin"])["dev"].max().to_frame()
        dfma = dfma.reset_index(drop=False).rename(
            {"index":"origin", "dev":"max_actual"}, axis=1)
        df = pd.merge(df, dfma, on="origin", how="outer", sort=False)
        df = df.sort_values(["origin", "dev"]).reset_index(drop=True)
        df["incl_actual"] = df["actual_ind"].map(lambda v: 1 if v==1 else 0)
        df["incl_pred"] = df.apply(
            lambda rec: 1 if (rec.actual_ind==0 or rec.dev==rec.max_actual) else 0,
            axis=1
            )

        # Vertically concatenate dfact and dfpred.
        dfact = df[df["incl_actual"]==1][["origin", "dev", "value"]]
        dfact["rectype"] = "actual"
        dfpred = df[df["incl_pred"]==1][["origin", "dev", "value"]]
        dfpred["rectype"] = "forecast"
        return(pd.concat([dfact, dfpred]).reset_index(drop=True))


    def plot(self, actuals_color="#334488", forecasts_color="#FFFFFF",
             axes_style="darkgrid", context="notebook", col_wrap=4,
             hue_kws=None, **kwargs):
        """
        Visualize actual losses along with projected chain ladder development.

        Parameters
        ----------
        actuals_color: str
            A color name or hexidecimal code used to represent actual
            observations. Defaults to "#00264C".

        forecasts_color: str
            A color name or hexidecimal code used to represent forecast
            observations. Defaults to "#FFFFFF".

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

        hue_kws: dictionary of param:list of values mapping
            Other keyword arguments to insert into the plotting call to let
            other plot attributes vary across levels of the hue variable
            (e.g. the markers in a scatterplot). Each list of values should
            have length 2, with each index representing aesthetic
            overrides for forecasts and actuals respectively. Defaults to
            ``None``.

        kwargs: dict
            Additional styling options for scatter points. This can override
            default values for ``plt.plot`` objects. For a demonstration,
            See the Examples section.

        Examples
        --------
        Demonstration of how to pass a dictionary of plot properties in order
        to update the scatter size and marker:

            In [1]: import trikit
            In [2]: raa = trikit.load(dataset="raa")
            In [3]: tri = trikit.totri(data=raa)
            In [4]: cl = tri.cl(sel="all-weighted", tail=1.005)

        ``cl`` represents an instance of ``ChainLadderResult``, which
        exposes the ``plot`` method. First, we compile the dictionary of
        attributes to override:

            In [6]: kwds = dict(marker="s", markersize=6)
            In [7]: cl.plot(**kwds)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_context(context)

        # Plot chain ladder projections by development period for each
        # origin year. FacetGrid's ``hue`` argument should be set to
        # "rectype".
        data = self._data_transform()
        with sns.axes_style(axes_style):
            huekwargs = dict(
                marker=["o", "o",], markersize=[6, 6,],
                color=["#000000", "#000000",], fillstyle=["full", "full",],
                markerfacecolor=[forecasts_color, actuals_color,],
                markeredgecolor=["#000000", "#000000",],
                markeredgewidth=[.50, .50,], linestyle=["-", "-",],
                linewidth=[.475, .475,],
                )

            if hue_kws is not None:
                # Determine whether the length of each element of hue_kws is 4.
                if all(len(hue_kws[i])==4 for i in hue_kws):
                    huekwargs.update(hue_kws)
                else:
                    warnings.warn("hue_kws overrides not correct length - Ignoring.")

            grid = sns.FacetGrid(
                data, col="origin", hue="rectype", hue_kws=huekwargs, col_wrap=col_wrap,
                margin_titles=False, despine=True, sharex=True, sharey=True,
                hue_order=["forecast", "actual",]
                )

            grid.map(plt.plot, "dev", "value",)
            grid.set_axis_labels("", "")
            grid.set(xticks=data.dev.unique().tolist())
            grid.set_titles("{col_name}", size=9)
            grid.set_xticklabels(data.dev.unique().tolist(), size=8)

            # Change ticklabel font size and place legend on each facet.
            for ii, _ in enumerate(grid.axes):
                ax_ = grid.axes[ii]
                legend_ = ax_.legend(
                    loc="lower right", fontsize="x-small", frameon=True,
                    fancybox=True, shadow=False, edgecolor="#909090",
                    framealpha=1, markerfirst=True,
                    )
                legend_.get_frame().set_facecolor("#FFFFFF")
                ylabelss = [jj.get_text() for jj in list(ax_.get_yticklabels())]
                ylabelsn = [float(jj.replace(u"\u2212", "-")) for jj in ylabelss]
                ylabelsn = [jj for jj in ylabelsn if jj>=0]
                ylabels = ["{:,.0f}".format(jj) for jj in ylabelsn]
                if (len(ylabels)>0):
                    ax_.set_yticklabels(ylabels, size=8)
                ax_.tick_params(
                    axis="x", which="both", bottom=True, top=False, labelbottom=True
                    )
                ax_.annotate(
                    origin_, xy=(.85, .925), xytext=(.85, .925), xycoords='axes fraction',
                    textcoords='axes fraction', fontsize=9, rotation=0, color="#000000",
                    )

                # Draw border around each facet.
                for _, spine in ax_.spines.items():
                    spine.set_visible(True)
                    spine.set_color("#000000")
                    spine.set_linewidth(.50)

        # Adjust facets downward and and left-align figure title.
        # plt.subplots_adjust(top=0.87)
        # grid.fig.suptitle(
        #     "Chain Ladder Ultimates by Origin", x=0.065, y=.975,
        #     fontsize=9, color="#404040", ha="left"
        #     )
        plt.show()


    def __str__(self):
        return(self.summary.to_string(formatters=self._summspecs))

    def __repr__(self):
        return(self.summary.to_string(formatters=self._summspecs))

