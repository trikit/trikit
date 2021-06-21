"""
This module contains the class definitions for ``BaseChainLadder``.-
"""
import collections
import functools
import warnings
import pandas as pd
import numpy as np



class BaseChainLadder:
    """
    From the Casualty Actuarial Society's "Estimating Unpaid Claims Using
    Basic Techniques" Version 3 (Friedland, Jacqueline - 2010), the
    development method ('Chain Ladder') consists of seven basic steps:

    1. Compile claims data in a development triangle.
    2. Calculate age-to-age factors.
    3. Calculate averages of the age-to-age factors.
    4. Select claim development factors.
    5. Select tail factor.
    6. Calculate cumulative claims.
    7. Project ultimate claims.

    The ``BaseChainLadder`` class encapsulates logic to perform steps 1-7.
    Other classes, including ``MackChainLadder and ``BoostrapChainLadder``
    inherit from ``BaseChainLadder``.

    References
    ----------
    1. Friedland, J., *Estimating Unpaid Claims Using Basic Techniques*,
       Casualty Actuarial Society, 2010.
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
        the application of the development technique. Returned object is an
        instance of ``BaseChainLadderResult``, which exposes a ``summary``
        attribute, a DataFrame with the following fields:

            * index: Origin period.

            * maturity: The age of the associated origin period in terms of
              development period duration.

            * cldf: Cumulative loss development factors.

            * emergence: 1 / cldf.

            * latest: The latest diagonal from the cumulative triangle instance.

            * ultimate: Projected ultimates. Computed as latest * cldf.

            * reserve: Chain ladder reserve estimates. Computed as
              ultimate - latest.

        Parameters
        ----------
        sel: str, pd.Series or array_like
            Either the ldf average description to select from
            ``self.tri.a2a_avgs``, or a pd.Series or array_like of LDFs to use
            in preference to LDFs computed from the triangle directly. For a
            triangle with n development periods, if ``sel`` is array_like and
            ``len(sel)==n``, the last entry is assumed to be the tail factor.
            If ``len(sel)==n-1``, then the ``tail`` parameter will be appended
            to the provided collection of loss development factors. Defaults
            to "all-weighted".

        tail: float
            Tail factor. Defaults to 1.0.

        Returns
        -------
        BaseChainLadderResult
        """
        if isinstance(sel, str):
            ldfs = self._ldfs(sel=sel, tail=tail)
        else:
            if isinstance(sel, pd.Series):
                # Check whether sel has the appropriate length.
                if not self.tri.devp.size==sel.index.size:
                    # Check if all but last development period are identical.
                    if not self.tri.devp[:-1].size==sel.index.size:
                        raise ValueError(
                            "sel has {} values, LDF overrides require at least {}.".format(
                                sel.size, self.tri.devp.size - 1
                                )
                            )
                    else:
                        # Only tail factor is missing: Append tail factor to sel.
                        increment = np.unique(sel.index[1:] - sel.index[:-1])[0]
                        sel.loc[sel.index.max() + increment] = tail

            elif isinstance(sel, (collections.Sequence, np.ndarray)):
                sel = np.asarray(sel, dtype=np.float)
                if sel.size!=self.tri.devp.size:
                    if sel.size==(self.tri.devp.size-1):
                        # Append sel with tail.
                        sel = np.append(sel, tail)
                    else:
                        raise ValueError(
                            "sel has {} values, LDF overrides require at least {}.".format(
                                sel.size, self.tri.devp.size - 1
                                )
                            )

            # Coerce sel to pd.Series.
            ldfs = pd.Series(sel, index=self.tri.devp, dtype=np.float)

        cldfs = self._cldfs(ldfs=ldfs)
        ultimates = self._ultimates(cldfs=cldfs)
        reserves = self._reserves(ultimates=ultimates)
        maturity = self.tri.maturity.astype(np.str)
        latest = self.tri.latest_by_origin
        trisqrd = self._trisqrd(ldfs=ldfs)

        # Compile chain ladder point estimate summary.
        dfmatur = maturity.to_frame().reset_index(drop=False).rename({"index":"origin"}, axis=1)
        dfcldfs = cldfs.to_frame().reset_index(drop=False).rename({"index":"maturity"}, axis=1)
        dfcldfs["maturity"] = dfcldfs["maturity"].astype(np.str)
        dfcldfs["emergence"] = 1 / dfcldfs["cldf"]
        dfsumm = dfmatur.merge(dfcldfs, on=["maturity"], how="left").set_index("origin")
        dfsumm.index.name = None
        dflatest = latest.to_frame().rename({"latest_by_origin":"latest"}, axis=1)
        dfsumm = functools.reduce(
            lambda df1, df2: df1.join(df2),
            (dflatest, ultimates.to_frame(), reserves.to_frame()), dfsumm
            )

        dfsumm.loc["total"] = dfsumm.sum()
        dfsumm.loc["total", "maturity"] = ""
        dfsumm.loc["total", ["cldf", "emergence"]] = np.NaN
        kwds = {"sel":sel,}

        # Initialize and return BaseChainLadderResult instance.
        cl_result = BaseChainLadderResult(
            summary=dfsumm, tri=self.tri, ldfs=ldfs, tail=tail, trisqrd=trisqrd, **kwds
            )

        return(cl_result)


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
        # Determine index for tail factor.
        ldfs = self.tri.a2a_avgs.loc[sel]
        increment = np.unique(ldfs.index[1:] - ldfs.index[:-1])[0]
        ldfs.loc[ldfs.index.max() + increment] = tail
        return(pd.Series(ldfs, name="ldf").sort_index())


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
        cldfs = np.cumprod(ldfs.values[::-1])[::-1]
        cldfs = pd.Series(data=cldfs, index=ldfs.index.values, name="cldf")
        return(cldfs.astype(np.float).sort_index())


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
        ultimates = pd.Series(
            data=self.tri.latest_by_origin.values * cldfs.values[::-1],
            index=self.tri.index, name="ultimate"
            )
        return(ultimates.astype(np.float).sort_index())


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
        reserves = pd.Series(
            data=ultimates - self.tri.latest_by_origin,
            index=self.tri.index, name='reserve')
        return(reserves.astype(np.float).sort_index())


    def _trisqrd(self, ldfs):
        """
        Project claims growth for each future development period. Returns a
        DataFrame of loss projections for each subsequent development period
        for each origin period. Populates the triangle's lower-right or
        southeast portion (i.e., the result of "squaring the triangle").

        Parameters
        ----------
        ldfs: pd.Series
            Selected ldfs, typically the output of calling ``self._ldfs``.

        Returns
        -------
        pd.DataFrame
        """
        trisqrd = self.tri.copy(deep=True)
        rposf = self.tri.index.size
        clvi = self.tri.clvi["row_offset"]
        for i in enumerate(trisqrd.columns[1:], start=1):
            ii  , devp  = i[0], i[1]
            ildf, rposi = ldfs.values[ii - 1], clvi[devp] + 1
            trisqrd.iloc[rposi:rposf, ii] = trisqrd.iloc[rposi:rposf, ii - 1] * ildf
        # Multiply right-most column by tail factor.
        max_devp = trisqrd.columns[-1]
        trisqrd["ultimate"] = trisqrd.loc[:,max_devp].values * ldfs.values[-1]
        return(trisqrd.astype(np.float).sort_index())



class BaseChainLadderResult:
    """
    Summary class consisting of output resulting from invocation of
    ``BaseChainLadder``'s ``__call__`` method.
    """
    def __init__(self, summary, tri, ldfs, tail, trisqrd, **kwargs):
        """
        Container object for ``BaseChainLadder`` output.

        Parameters
        ----------
        summary: pd.DataFrame
            Chain Ladder summary compilation.

        tri: trikit.triangle._CumTriangle
            A cumulative triangle instance.

        tail: float
            Tail factor. Defaults to 1.0.

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
        self.emergence = summary["emergence"]
        self.ultimates = summary["ultimate"]
        self.maturity = summary["maturity"]
        self.reserves = summary["reserve"]
        self.latest = summary["latest"]
        self.cldfs = summary["cldf"]
        self.summary = summary
        self.trisqrd = trisqrd
        self.ldfs = ldfs
        self.tail = tail
        self.tri = tri

        if kwargs is not None:
            for key_ in kwargs:
                setattr(self, key_, kwargs[key_])

        self._summspecs = {
            "ultimate":"{:,.0f}".format, "reserve":"{:,.0f}".format,
            "latest":"{:,.0f}".format, "cldf":"{:.5f}".format,
            "emergence":"{:.5f}".format,
            }


    @staticmethod
    def _get_yticks(x):
        """
        Determine y axis tick labels for a given maximum loss amount x.
        Return tuple of tick values and ticklabels.

        Parameters
        ----------
        x: float
            Maximum value for a given origin period.

        Returns
        -------
        tuple of ndarrays
        """
        ref_divs = np.power(10, np.arange(10))
        div_index = np.where((x / ref_divs) > 1)[0].max()
        x_div = ref_divs[div_index]

        # Find upper limit for y-axis given origin_max_val.
        yuls_seq = x_div * np.arange(1, 11)
        x_yuls = yuls_seq - x
        yul = yuls_seq[np.where(x_yuls>0)[0].min()]
        y_ticks = np.linspace(0, yul, num=5)
        y_ticklabels = np.asarray(["{:,.0f}".format(ii) for ii in y_ticks])
        return(y_ticks, y_ticklabels)


    def _data_transform(self):
        """
        Transform dataset for use in FacetGrid plot by origin exhibting chain
        ladder reserve estimates.

        Returns
        -------
        pd.DataFrame
        """
        trisqrd = self.trisqrd.reset_index(drop=False).rename({"index":"origin" }, axis=1)
        df0 = pd.melt(trisqrd, id_vars=["origin"], var_name="dev", value_name="value")

        # Combine df0 with latest cumulative loss by origin period.
        df0 = df0.merge(
            self.latest.reset_index(drop=False).rename({"index":"origin"}, axis=1),
            on="origin", how="left"
            )

        dfult = df0[df0["dev"]=="ultimate"]
        dev_increment = np.unique(self.ldfs.index[1:] - self.ldfs.index[:-1])[0]
        dfult["dev"] =  self.ldfs.index.max() + dev_increment
        dfult["rectype"] = "forecast"
        df0 = df0[df0["dev"]!="ultimate"].reset_index(drop=True)

        # Create tabular dataset based on tri.triind. Rows wiith 0s represent
        # actuals, rows with 1 represent forecasts.
        df1 = self.tri.triind.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        df1 = pd.melt(df1, id_vars=["origin"], var_name="dev", value_name="value")
        df1["value"] = df1["value"].map(lambda v: 1 if v==0 else 0)
        df1 = df1[~np.isnan(df1["value"])].rename(
            {"value":"actual_ind"}, axis=1).reset_index(drop=True)

        # Combine df0 and df1 into a single DataFrame, then perform cleanup
        # actions for cases in which df0 has more records than df1.
        df = pd.merge(df0, df1, on=["origin", "dev"], how="left", sort=False)

        # Bind reference to maximum dev period for each origin.
        dfma = df[df["actual_ind"]==1].groupby(
            ["origin"])["dev"].max().to_frame().reset_index(drop=False).rename(
            {"index":"origin", "dev":"max_actual"}, axis=1
            )
        df = pd.merge(df, dfma, on="origin", how="left", sort=False)
        df["incl_actual"] = df["actual_ind"].map(lambda v: 1 if v==1 else 0)
        df["incl_pred"] = df.apply(
            lambda rec: 1 if (rec.actual_ind==0 or rec.dev==rec.max_actual) else 0,
            axis=1
            )

        # Split data into actual and pred cohorts, then recombine. Note that
        # the latest cumulative loss by origin intentionally appears in both
        # datasets.
        dfact = df[df["incl_actual"]==1][["origin", "dev", "value", "latest"]]
        dfact["rectype"] = "actual"
        dfpred = df[df["incl_pred"]==1][["origin", "dev", "value", "latest"]]
        dfpred["rectype"] = "forecast"

        # Create total DataFrame, representing losses across all origin periods
        # by development period and at ultimate.
        dftotal = pd.concat([
            dfpred.groupby(["dev", "rectype"], as_index=False)[["value", "latest"]].sum(),
            dfult.groupby(["dev", "rectype"], as_index=False)[["value", "latest"]].sum()
            ])

        # Combine dfact, dfpred, dfult and dftotal.
        dftotal["origin"] = "total"
        dfall = pd.concat([dfact, dfpred, dfult, dftotal]).reset_index(drop=True).rename(
            {"value":"loss"}, axis=1
            )


        # Add origin index column sort origin columns, which is of type object
        # # since adding "total".
        dfall["dev"] = dfall["dev"].astype(np.int)
        origin_vals = sorted([int(ii) for ii in dfall["origin"].unique() if ii!="total"])
        dindex = {jj:ii for ii,jj in enumerate(origin_vals)}
        dindex.update({"total":max(dindex.values())+1})
        dfall["origin_index"] = dfall["origin"].map(dindex)

        # Add reserve column, defined as value - latest when rectype=="forecast",
        # otherwise 0.
        dfall["reserve"] = dfall.apply(
            lambda rec: rec.loss - rec.latest if rec.rectype=="forecast" else 0,
            axis=1
            )
        column_order = ["origin_index", "origin", "dev", "loss", "reserve", "rectype",]
        return(dfall[column_order].reset_index(drop=True))


    def plot(self, actuals_color="#334488", forecasts_color="#FFFFFF", axes_style="darkgrid",
             context="notebook", col_wrap=4, hue_kws=None, exhibit_path=None, **kwargs):
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

        exhibit_path: str
            Path to which exhibit should be written. If None, exhibit will be
            rendered via ``plt.show()``.

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

        ``cl`` represents an instance of ``BaseChainLadderResult``, which
        exposes the ``plot`` method. First, we compile the dictionary of
        attributes to override:

            In [6]: kwds = dict(marker="s", markersize=6)
            In [7]: cl.plot(**kwds)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_context(context)

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
                data, col="origin", hue="rectype", hue_kws=huekwargs,
                col_wrap=col_wrap, margin_titles=False, despine=True, sharex=False,
                sharey=False, hue_order=["forecast", "actual",]
                )

            ult_vals = grid.map(plt.plot, "dev", "loss",)
            devp_xticks = np.sort(data.dev.unique())
            devp_xticks_str = [
                str(ii) if ii!=devp_xticks.max() else "ult" for ii in devp_xticks
                ]
            grid.set(xticks=devp_xticks);  grid.set_xticklabels(devp_xticks_str, size=7)
            origin_order = data[["origin_index", "origin"]].drop_duplicates().sort_values(
                "origin_index").origin.values

            with warnings.catch_warnings():

                warnings.simplefilter("ignore")

                for origin, ax_ii in zip(origin_order, grid.axes):

                    legend = ax_ii.legend(
                        loc="lower right", fontsize="x-small", frameon=True,
                        fancybox=True, shadow=False, edgecolor="#909090",
                        framealpha=1, markerfirst=True,
                        )
                    legend.get_frame().set_facecolor("#FFFFFF")

                    # For given origin, determine optimal 5-point tick labels.
                    origin_max_val = data[data.origin==origin].loss.max()
                    y_ticks, y_ticklabels = self._get_yticks(origin_max_val)
                    ax_ii.set_yticks(y_ticks)
                    ax_ii.set_yticklabels(y_ticklabels, size=7)
                    ax_ii.annotate(
                        origin, xy=(.075, .90), xytext=(.075, .90), xycoords='axes fraction',
                        textcoords='axes fraction', fontsize=9, rotation=0, color="#000000",
                        )
                    ax_ii.set_title(""); ax_ii.set_xlabel(""); ax_ii.set_ylabel("")

                    # Draw border around each facet.
                    for _, spine in ax_ii.spines.items():
                        spine.set(visible=True, color="#000000", linewidth=.50)

            if exhibit_path is not None:
                plt.savefig(exhibit_path)
            else:
                plt.show()


    def __str__(self):
        return(self.summary.to_string(formatters=self._summspecs))


    def __repr__(self):
        return(self.summary.to_string(formatters=self._summspecs))

