"""
This module contains the definitions of both the ``_IncrTriangle`` and
``_CumTriangle`` classes. Users should avoid instantiating ``_IncrTriangle``
or ``_CumTriangle`` instances directly; rather the dataset and triangle
arguments should be passed to ``totri``, which will return either an
instance of ``_CumTriangle`` or ``_IncrTriangle``, depending on the argument
specified for ``type_``.

##### TODO #####
[0] Refactor a2a_avgs
[1] Integrate a2aind into a2a_avgs
[2] Make a2a_avgs a function rather than a property.
"""
import itertools
import numpy as np
import pandas as pd


from .chainladder import _BaseChainLadder
#from .chainladder import bootstrap._BootstrapChainLadder
from .chainladder.bootstrap import _BootstrapChainLadder
from .chainladder.mack import _MackChainLadder
# from chainladder.bootstrap import _BootstrapChainLadder


class _IncrTriangle(pd.DataFrame):
    """
    ``_IncrTriangle`` specification. Note that this class isn't part of
    trikit's public interface, as is used  primarily as a base class for
    ``_CumTriangle`` object definitions. As such, users shouldn't
    instantiate ``_IncrTriangle`` instances directly. To obtain an
    incremental triangle object, use ``trikit.totri``.
    """
    def __init__(self, data, origin=None, dev=None, value=None):
        """
        Attempts to convert ``data`` to an incremental triangle instance.
        Checks for ``origin``, ``dev`` and ``value`` arguments for fieldnames
        corresponding to origin year, development period and loss amount
        respectively. If ``origin``, ``dev`` and ``value`` are unspecified,
        fieldnames are assumed to be "origin", "dev" and "value".

        Parameters
        ----------
        data: pd.DataFrame
            The dataset to be coerced into a ``_IncrTriangle`` instance.
            ``data`` must be tabular loss data with at minimum columns
            representing the origin/acident year, the development
            period and the actual loss amount, given by ``origin``, ``dev``
            and ``value`` arguments.

        origin: str
            The fieldname in ``data`` representing the origin year.
            Defaults to None.

        dev: str
            The fieldname in ``data`` representing the development period.
            Defaults to None.

        value: str
            The fieldname in ``data`` representing loss amounts.
            Defaults to None.

        Returns
        -------
        trikit.triangle._IncrTriangle
        """
        try:
            if all(i is None for i in (origin, dev, value)):
                origin, dev, value = "origin", "dev", "value"
            data2 = data[[origin, dev, value]]
            data2 = data2.groupby([origin, dev], as_index=False).sum()
            tri = data2.pivot(index=origin, columns=dev).rename_axis(None)
            tri.columns = tri.columns.droplevel(0)

        except KeyError:
            print("One or more fields not present in data.")

        # Force all triangle cells to be of type np.float_.
        for i in tri:
            tri[i] = tri[i].astype(np.float_)

        tri.columns.name = None

        super().__init__(tri)

        self.origin = origin
        self.value = value
        self.dev = dev

        # Properties.
        self._latest_by_origin = None
        self._latest_by_devp = None
        self._nbr_cells = None
        self._maturity = None
        self._triind = None
        self._devp = None
        self._latest = None
        self._origins = None
        self._rlvi = None
        self._clvi = None
        self._dof = None



    @property
    def nbr_cells(self):
        """
        Return the number of non-NaN cells.

        Returns
        -------
        int
        """
        if self._nbr_cells is None:
            self._nbr_cells = self.count().sum()
        return(self._nbr_cells)


    @property
    def dof(self):
        """
        Return the degress of freedom.

        Returns
        -------
        int
        """
        if self._dof is None:
            self._dof = self.nbr_cells - (self.columns.size-1) + self.index.size
        return(self._dof)


    @property
    def triind(self):
        """
        Table indicating forecast cells with 1, actual data with 0.

        Returns
        -------
        pd.DataFrame
        """
        if self._triind is None:
            self._triind = self.applymap(lambda x: 1 if np.isnan(x) else 0)
        return(self._triind)


    @property
    def rlvi(self):
        """
        Determine the last valid index by row/origin period.

        Returns
        -------
        pd.DataFrame
        """
        if self._rlvi is None:
            self._rlvi = pd.DataFrame({
                "dev":self.apply(
                    lambda x: x.last_valid_index(), axis=1).values
                },index=self.index)
            self._rlvi["col_offset"] = \
                self._rlvi["dev"].map(lambda x: self.columns.get_loc(x))
        return(self._rlvi)


    @property
    def clvi(self):
        """
        Determine the last valid index by development period.

        Returns
        -------
        pd.DataFrame
        """
        if self._clvi is None:
            self._clvi = pd.DataFrame({
                "origin":self.apply(lambda x: x.last_valid_index(), axis=0).values
                },index=self.columns)
            self._clvi["row_offset"] = \
                self._clvi["origin"].map(lambda x: self.index.get_loc(x))
        return(self._clvi)


    @property
    def latest(self):
        """
        Return the values on the triangle's latest diagonal. Loss amounts
        are given, along with the associated origin year and development
        period. The latest loss amount by origin year alone can be  obtained
        by calling ``self.latest_by_origin``, or by development period by
        calling by ``self.latest_by_devp``.

        Returns
        -------
        pd.DataFrame
        """
        if self._latest is None:
            lindx = self.apply(lambda dev_: dev_.last_valid_index(), axis=1)
            self._latest = pd.DataFrame(
                {"latest":self.lookup(lindx.index, lindx.values),
                 "origin":lindx.index, "dev":lindx.values})
        return(self._latest[["origin", "dev", "latest"]].sort_index())


    @property
    def latest_by_origin(self):
        """
        Return the latest loss amounts by origin year.

        Returns
        -------
        pd.Series
        """
        if self._latest_by_origin is None:
            self._latest_by_origin = pd.Series(
                data=self.latest["latest"].values, index=self.latest["origin"].values,
                name="latest_by_origin")
        return(self._latest_by_origin.sort_index())


    @property
    def latest_by_devp(self):
        """
        Return the latest loss amounts by development period.

        Returns
        -------
        pd.Series
        """
        if self._latest_by_devp is None:
            self._latest_by_devp = pd.Series(
                data=self.latest["latest"].values, index=self.latest["dev"].values,
                name="latest_by_devp")
        return(self._latest_by_devp.sort_index())


    @property
    def devp(self):
        """
        Return triangle's development periods.

        Returns
        -------
        pd.Series
        """
        if self._devp is None:
            self._devp = pd.Series(self.columns,name="dev")
        return(self._devp.sort_index())


    @property
    def origins(self):
        """
        Return triangle's origin periods.

        Returns
        -------
        pd.Series
        """
        if self._origins is None:
            self._origins = pd.Series(self.index, name="origin")
        return(self._origins.sort_index())


    @property
    def maturity(self):
        """
        Return the maturity for each origin period.

        Returns
        -------
        ps.Series
        """
        if self._maturity is None:
            dfind, matlist  = (1 - self.triind), list()
            for i in range(dfind.index.size):
                lossyear = dfind.index[i]
                maxindex = dfind.loc[lossyear].to_numpy().nonzero()[0].max()
                itermatur = dfind.columns[maxindex]
                matlist.append(itermatur)
            self._maturity = pd.Series(data=matlist, index=self.index, name="maturity")
        return(self._maturity.sort_index())


    def as_tbl(self):
        """
        Transform triangle instance into a tabular representation.

        Returns
        -------
        pd.DataFrame
        """
        tri_ = self.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        df_ = pd.melt(tri_, id_vars=[self.origin], var_name=self.dev, value_name=self.value)
        df_ = df_[~np.isnan(df_[self.value])]
        df_ = df_.astype({self.origin:np.int_, self.dev:np.int_, self.value:np.float_})
        return(df_.sort_values(by=[self.origin, self.dev]).reset_index(drop=True))


    def as_cum(self):
        """
        Transform incremental triangle instance into a cumulative
        representation. Note that returned object will be a DataFrame,
        not an instance of ``triangle._CumTriangle``.

        Returns
        -------
        pd.DataFrame
        """
        return(self.cumsum(axis=1))


    def as_incr(self):
        """
        Transform ``triangle._IncrTriangle`` instance to pd.DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return(pd.DataFrame(self))


    # def __str__(self):
    #     # Controls when print(self) is called
    #     formats_ = {dev_:"{:.0f}".format for dev_ in self.columns}
    #     return(self.to_string(formatters=formats_))


    def __repr__(self):
        """
        Controls when object is referenced from interpreter.
        """
        formats_ = {dev_:"{:.0f}".format for dev_ in self.columns}
        return(self.to_string(formatters=formats_))





class _CumTriangle(_IncrTriangle):
    """
    Cumulative triangle class definition.
    """
    def __init__(self, data, origin=None, dev=None, value=None):
        """
        Attempts to represent ``data`` as a cumulative triangle instance.
        ``origin``, ``dev`` and ``value`` arguments represent fieldnames
        corresponding to loss year, development period and loss amount
        respectively. If ``origin``, ``dev`` and ``value`` are unspecified,
        fieldnames are assumed to be "origin", "dev" and "value".

        Parameters
        ----------
        origin: str
            The fieldname in ``data`` representing the origin year.
            Defaults to None.

        dev: str
            The fieldname in ``data`` representing the development period.
            Defaults to None.

        value: str
            The fieldname in ``data`` representing loss amounts.
            Defaults to None.

        Returns
        -------
        trikit.triangle._CumTriangle
        """
        if all(i is None for i in (origin, dev, value)):
            origin, dev, value = "origin", "dev", "value"
        data2 = data[[origin, dev, value]]
        data2 = data2.groupby([origin, dev], as_index=False).sum()
        data2 = data2.sort_values(by=[origin, dev])
        data2["cumval"] = data2.groupby([origin], as_index=False)[value].cumsum()
        data2 = data2.drop(value, axis=1)
        data2 = data2.rename(columns={"cumval":value})
        super().__init__(data=data2, origin=origin, dev=dev, value=value)

        # Properties
        self._a2a_avgs = None
        self._a2aind = None
        self._a2a = None


    @staticmethod
    def _geometric(vals, weights=None):
        """
        Compute the geometric average of the elements of ``vals``.

        Parameters
        ----------
        vals: np.ndarray
            An array of values, typically representing link ratios from a
            single development period.

        weights: np.ndarray
            Weights to assign specific values in the average computation.
            If None, each value is assigned equal weight.

        Returns
        -------
        float
        """
        if len(vals)==0:
            avg_ = None
        else:
            avg_ = np.prod(vals) ** (1 / len(vals))
        return(avg_)


    @staticmethod
    def _simple(vals, weights=None):
        """
        Compute the simple average of elements of ``vals``.

        Parameters
        ----------
        vals: np.ndarray
            An array of values, typically representing link ratios from a
            single development period.

        weights: np.ndarray
            Weights to assign specific values in the average computation.
            If None, each value is assigned equal weight.

        Returns
        -------
        float
        """
        if len(vals) < 1:
            avg_ = None
        else:
            avg_ = sum(vals) / len(vals)
        return(avg_)


    @staticmethod
    def _medial(vals, weights=None):
        """
        Compute the medial average of elements in ``vals``. Medial average
        eliminates the min and max values, then returns the arithmetic
        average of the remaining items.

        Parameters
        ----------
        vals: np.ndarray
            An array of values, typically representing link ratios from a
            single development period.

        weights: np.ndarray
            Weights to assign specific values in the average computation.
            If None, each value is assigned equal weight.

        Returns
        -------
        float
        """
        vals = list(vals)
        if len(vals)==0:
            avg_ = None
        elif len(vals)==1:
            avg_ = vals[0]
        elif len(vals)==2:
            avg_ = sum(vals) / len(vals)
        else:
            keep_ = sorted(vals)[1:-1]
            avg_ = sum(keep_) / len(keep_)
        return(avg_)


    @property
    def a2a(self):
        """
        Compute adjacent proportions, a.k.a. link ratios.

        Returns
        -------
        pd.DataFrame
        """
        if self._a2a is None:
            self._a2a = self.shift(periods=-1, axis=1) / self
            self._a2a = self._a2a.dropna(axis=1, how="all").dropna(axis=0, how="all")
        return(self._a2a.sort_index())


    @property
    def a2aind(self):
        """
        Determine which cells should be included and which to exclude
        when computing age-to-age averages.

        Returns
        -------
        pd.DataFrame
        """
        if self._a2aind is None:
            self._a2aind = self.a2a.applymap(lambda v: 0 if np.isnan(v) else 1)
        return(self._a2aind)


    @a2aind.setter
    def a2aind(self, update_spec):
        """
        Update ``self.a2aind`` in order to down-weight ldfs in Chain Ladder
        calculation.

        Parameters
        ----------
        update_spec: tuple
            3-tuple consisting of ``(index, column, value)``, representing
            the intersection point of the ``self.a2a`` target cell, and the
            value used to update it.

        Examples
        --------
        Load raa sample dataset, and remove a highly-leveraged age-to-age
        factor from influencing the ldf calculation.

        >>> import trikit
        >>> raa = trikit.load(dataset="raa")
        >>> tri = trikit.totri(data=raa)
        >>> tri.a2a.iloc[:, :1]
                      1
        1981   1.649840
        1982  40.424528
        1983   2.636950
        1984   2.043324
        1985   8.759158
        1986   4.259749
        1987   7.217235
        1988   5.142117
        1989   1.721992

        To remove the link ratio at origin year 1982 and development
        period 1, run the following:

        >>> tri.a2aind = (1982, 1, 0)
        >>> tri.a2aind
              1  2  3  4  5  6  7  8  9
        1981  1  1  1  1  1  1  1  1  1
        1982  0  1  1  1  1  1  1  1  0
        1983  1  1  1  1  1  1  1  0  0
        1984  1  1  1  1  1  1  0  0  0
        1985  1  1  1  1  1  0  0  0  0
        1986  1  1  1  1  0  0  0  0  0
        1987  1  1  1  0  0  0  0  0  0
        1988  1  1  0  0  0  0  0  0  0
        1989  1  0  0  0  0  0  0  0  0

        Notice that the value at (1982, 1) is 0. To change it back
        to 1, simply run:

        >>> tri.a2aind = (1982, 1, 1)
        >>> tri.a2aind
              1  2  3  4  5  6  7  8  9
        1981  1  1  1  1  1  1  1  1  1
        1982  1  1  1  1  1  1  1  1  0
        1983  1  1  1  1  1  1  1  0  0
        1984  1  1  1  1  1  1  0  0  0
        1985  1  1  1  1  1  0  0  0  0
        1986  1  1  1  1  0  0  0  0  0
        1987  1  1  1  0  0  0  0  0  0
        1988  1  1  0  0  0  0  0  0  0
        1989  1  0  0  0  0  0  0  0  0

        Note also that ``self.a2aind`` may be updated using DataFrame
        methods directly:

        >>> tri.a2aind.at[1982, 1] = 0
        >>> tri.a2aind
              1  2  3  4  5  6  7  8  9
        1981  1  1  1  1  1  1  1  1  1
        1982  0  1  1  1  1  1  1  1  0
        1983  1  1  1  1  1  1  1  0  0
        1984  1  1  1  1  1  1  0  0  0
        1985  1  1  1  1  1  0  0  0  0
        1986  1  1  1  1  0  0  0  0  0
        1987  1  1  1  0  0  0  0  0  0
        1988  1  1  0  0  0  0  0  0  0
        1989  1  0  0  0  0  0  0  0  0
        """
        indx, column, value = update_spec
        self._a2aind.at[indx, column] = value


    @property
    def a2a_avgs(self):
        """
        Compute age-to-age factors based on ``self.a2a`` table of adjacent
        proportions. Averages computed include "simple", "geometric", "medial"
        and "weighted".

        Returns
        -------
        pd.DataFrame
        """
        if self._a2a_avgs is None:

            _nbr_periods = list(range(1, self.a2a.shape[0])) + [0]
            indxstrs = list()

            # Create lookup table for average functions.
            avgfuncs = {
                'simple'   :self._simple,
                'geometric':self._geometric,
                'medial'   :self._medial,
                'weighted' :None
                }

            # Remove `0` entry, and add as last element of list.
            ldf_avg_lst = list(itertools.product(avgfuncs.keys(), _nbr_periods))

            indxstrs = [
                "all-" + str(i[0]) if i[1]==0 else "{}-{}".format(i[0], i[1])
                    for i in ldf_avg_lst
                ]

            # for i in ldf_avg_lst:
            #     iteravg, iterdur = i[0], i[1]
            #     iterstr = "all-" + str(iteravg) if iterdur==0 \
            #               else str(iterdur) + "-" + str(iteravg)
            #     indxstrs.append(iterstr)

            indx = sorted(ldf_avg_lst, key=lambda x: x[1])
            self._a2a_avgs = pd.DataFrame(index=indxstrs, columns=self.a2a.columns)

            for a in enumerate(ldf_avg_lst):
                duration, avgtype, indxpos = a[1][1], a[1][0], a[0]
                indxstr, iterfunc = indxstrs[indxpos], avgfuncs[avgtype]
                for col in range(self.a2a.shape[1]):
                    itercol, colstr = self.a2a.iloc[:, col], self.a2a.columns[col]

                    if avgtype=='weighted':
                        t_ic_1, t_ic_2 = self.iloc[:, col], self.iloc[:, (col + 1)]
                        # Find first NaN value in t_ic_2.
                        first_nan_year = t_ic_2.index[t_ic_2.count():][0]
                        first_nan_indx = t_ic_2.index.searchsorted(first_nan_year)
                        final_cell_indx = first_nan_indx
                        if duration==0:
                            first_cell_indx = 0
                        else:
                            first_cell_indx = (final_cell_indx-duration) if \
                                              (final_cell_indx-duration)>=0 else 0

                        # Divide sum of t_ic_2 by t_ic_1.
                        ic_2 = t_ic_2[first_cell_indx:final_cell_indx]
                        ic_1 = t_ic_1[first_cell_indx:final_cell_indx]
                        sum_ic_2 = t_ic_2[first_cell_indx:final_cell_indx].sum()
                        sum_ic_1 = t_ic_1[first_cell_indx:final_cell_indx].sum()

                        try:
                            iteravg = (sum_ic_2/sum_ic_1)
                        except ZeroDivisionError:
                            iteravg = np.NaN

                    else: # avgtype in ('simple', 'geometric', 'medial')
                        # find index of first row with NaN
                        if any(itercol.map(lambda x: np.isnan(x))):
                            first_nan_year = \
                                itercol.index[itercol.apply(lambda x: np.isnan(x))][0]
                            first_nan_indx = \
                                itercol.index.searchsorted(first_nan_year)
                            final_cell_indx = first_nan_indx
                            if duration==0:
                                first_cell_indx = 0
                            else:
                                first_cell_indx = (final_cell_indx-duration) if \
                                                  (final_cell_indx-duration)>=0 else 0

                        else: # itercol has 0 NaN's
                            final_cell_indx = len(itercol)
                            first_cell_indx = 0 if duration==0 else (final_cell_indx-duration)
                        try:
                            iteravg = iterfunc(itercol[first_cell_indx:final_cell_indx])

                        except ZeroDivisionError:
                            iteravg = np.NaN

                    self._a2a_avgs.loc[indxstr, colstr] = iteravg

        return(self._a2a_avgs)


    def plot(self, color="#334488", axes_style="darkgrid", context="notebook",
             col_wrap=5, **kwargs):
        """
        Visualize triangle development patterns.

        Parameters
        ----------
        facets: bool
            If True, loss development plots for each origin year are faceted,
            otherwise development patterns are plotted together on a single set
            of axes.

        axes_style: str
            Aesthetic style of plots. Defaults to "darkgrid". Other options
            include: {whitegrid, dark, white, ticks}.

        context: str
            Set the plotting context parameters. According to the seaborn
            documentation, This affects things like the size of the labels,
            lines, and other elements of the plot, but not the overall style.
            Defaults to ``"notebook"``. Additional options include
            {paper, talk, poster}.

        palette: str
            Selected matplotlib color map. For additional options, visit:
            https://matplotlib.org/tutorials/colors/colormaps.html.

        kwargs: dict
            Additional plot styling options.

        Returns
        -------
        matplotlib.pyplot.plot
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_context(context)
        data = self.as_tbl()

        with sns.axes_style(axes_style):

            pltkwargs = dict(
                marker="o", markersize=7, alpha=1, markeredgecolor="#000000",
                markeredgewidth=.50, linestyle="--", linewidth=.75,
                fillstyle="full", color=color,
                )

            if kwargs:
                pltkwargs.update(kwargs)

            titlestr_ = "Actual Loss Development by Origin"

            g = sns.FacetGrid(
                data, col="origin", col_wrap=col_wrap, margin_titles=False,
                despine=True, sharex=True, sharey=True,
                )

            g.map(plt.plot, "dev", "value", **pltkwargs)
            g.set_axis_labels("", "")
            g.set(xticks=data.dev.unique().tolist())
            g.set_titles("{col_name}", size=9)
            g.set_xticklabels(data.dev.unique().tolist(), size=8)

            # Change ticklabel font size and place legend on each facet.
            for i, _ in enumerate(g.axes):
                ax_ = g.axes[i]
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


    def as_incr(self):
        """
        Convert cumulative triangle instance into an incremental
        representation. Note that returned object will not be of type
        ``triangle._IncrTriangle``, but will instead be a DataFrame.

        Returns
        -------
        ps.DataFrame
        """
        tri_ = self.diff(axis=1)
        tri_.iloc[:, 0] = self.iloc[:, 0]
        return(tri_)


    def as_cum(self):
        """
        Transform ``triangle._CumTriangle`` instance to pd.DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return(pd.DataFrame(self))


    def chladder(self, range_method=None, **kwargs):
        """
        Apply Chain Ladder method to ``CumTriangle`` instance.

        Parameters
        ----------
        range_method: {"bootstrap", "mack", "mcmc"}
            Specifies the method to use to quantify ultimate/reserve
            variability. When ``range_method=None``, reduces to the standard
            chain ladder technique providing point estimate reserve
            projections at ultimate. Defaults to None.

        kwargs: dict
            For each value of ``range_method``, there are a number of optional
            parameters that can be used to override the default behavior of
            the algorithm in question. If a keyword argument is provided that
            is not valid within the context of the given Chain Ladder variant,
            the argument will be ignored and a warning will be generated.
            What follows are valid optional keyword parameters that can be
            passed into ``chladder`` for different values of ``range_method``:

            * ``range_method==None`` (Standard chain ladder)
                - ``sel``: The ldf average to select from ``triangle._CumTriangle.a2a_avgs``.
                Defaults to "all-weighted".
                - ``tail``: Tail factor. Defaults to 1.0.

            * ``range_method="bootstrap"`` (Bootstrap chain ladder)
                - ``sims``: The number of bootstrap resamplings to perform.
                Defaults to 1000.
                - ``q``: Determines which percentiles of the reserve distribution
                to compute. Defaults to [.75, .95].
                - ``neg_handler``: Dictates how negative triangle values should
                be handled. See documentation for ``_BoostrapChainLadder``
                for more information. Defaults to 1.
                - ``procdist``: The distribution used to incorporate process
                variance. Currently, this can only be set to "gamma".
                - ``parametric``: If True, fit standardized residuals to a
                normal distribution then sample from this parameterized
                distribution. Otherwise, sample with replacement from the
                collection of standardized residuals. Defaults to False.
                - ``symmetric``: Whether the symmetric interval of given
                ``q``('s) should be included in summary output.
                - ``interpolation``: See ``numpy.quantile`` for more information.
                Defaults to "linear".
                - ``random_state``: Set random seed for reproducibility.
                Defaults to None.

            * ``range_method="mack"`` (Mack chain ladder)
                - ``alpha``: Can be one of {0, 1, 2}. See ``_MackChainLadder._ldfs``
                for more information. Defaults to 1.
                - ``q``: Determines which percentiles of the reserve distribution
                to compute. Defaults to [.75, .95].
                - ``symmetric``: Whether the symmetric interval of given
                ``q``('s) should be included in summary output.

            * ``range_method="mcmc"`` (not yet implemented)
                - ``q``: Determines which percentiles of the reserve distribution
                to compute. Defaults to [.75, .95].
                - ``symmetric``: Whether the symmetric interval of given
                ``q``('s) should be included in summary output.

        Returns
        -------
        chainladder.*ChainLadderResult
            One of {``_BaseChainLadderResult``, ``_BootstrapChainLadderResult``,
                    ``_MackChainLadderResult``, ``_MCMCChainLadderResult``}

        Examples
        --------
        In the following examples we refer to the raa sample dataset which
        can be retrieved as follows:

        >>> import trikit
        >>> RAA = trikit.load("raa")
        >>> tri = trikit.totri(RAA)


        1. Perform standard chain ladder technique, overriding ``sel`` and ``tail``:

        >>> kwds = dict(sel="medial-5", tail=1.015)
        >>> cl = tri.chladder(**kwds)

        2. Perform boostrap chain ladder, overriding ``sims``, ``q`` and ``symmetric``:

        >>> kwds = dict(sims=2500, q=[.90, .99], symmetric=True)
        >>> bcl = tri.chladder(range_method="bootstrap", **kwds)

        3. Perfrom Mack chain ladder, overriding ``alpha``:

        >>> kwds = {"alpha":2}
        >>> mcl = tri.chladder(range_method="mack", **kwds)
        """
        kwds = {} if kwargs is None else kwargs
        if range_method is None:
            cl_ = _BaseChainLadder(self).__call__(**kwds)
        elif range_method.lower().strip().startswith("boot"):
            cl_ = _BootstrapChainLadder(self).__call__(**kwds)
        elif range_method.lower().strip().startswith("mack"):
            cl_ = _MackChainLadder(self).__call__(**kwds)
        elif range_method.lower().startswith("mcmc"):
            raise NotImplementedError("range_method='mcmc' not currently available.")
        else:
            raise ValueError("Invalid range_method specification: {}".format(range_method))
        return(cl_)
