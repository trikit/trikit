"""
This module contains the definitions of both the ``IncrTriangle`` and
``CumTriangle`` classes. Users should avoid instantiating ``IncrTriangle``
or ``CumTriangle`` instances directly; rather the dataset and triangle
arguments should be passed to ``totri``, which will return either an
instance of ``CumTriangle`` or ``IncrTriangle``, depending on the argument
specified for ``type_``.
"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from .chainladder import BaseChainLadder
from .chainladder.bootstrap import BootstrapChainLadder





class _BaseTriangle(pd.DataFrame):

    def __init__(self, data, origin=None, dev=None, value=None):
        """
        Transforms ``data`` into a triangle instance.

        Parameters
        ----------
        data: pd.DataFrame
            The dataset to be transformed into a ``_BaseTriangle`` instance.
            ``data`` must be tabular loss data with at minimum columns
            representing the origin/acident year, the development
            period and the actual loss amount, given by ``origin``, ``dev``
            and ``value`` arguments.

        origin: str
            The fieldname in ``data`` representing origin year.

        dev: str
            The fieldname in ``data`` representing development period.

        value: str
            The fieldname in ``data`` representing loss amounts.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be an instance of pd.DataFrame.")

        origin_ = "origin" if origin is None else origin
        if origin_ not in data.columns:
            raise KeyError("`{}` not present in data.".format(origin_))

        dev_ = "dev" if dev is None else dev
        if dev_ not in data.columns:
            raise KeyError("`{}` not present in data.".format(dev_))

        value_ = "value" if value is None else value
        if value_ not in data.columns:
            raise KeyError("`{}` not present in data.".format(value_))

        data2 = data.copy(deep=True)
        data2 = data2[[origin_, dev_, value_]]
        data2 = data2.groupby([origin_, dev_], as_index=False).sum()
        data2 = data2.sort_values(by=[origin_, dev_])
        tri = data2.pivot(index=origin_, columns=dev_).rename_axis(None)
        tri.columns = tri.columns.droplevel(0)

        # Force all triangle cells to be of type np.float.
        tri = tri.astype({kk:np.float for kk in tri.columns})
        tri.columns.name = None

        super().__init__(tri)

        self.origin = origin_
        self.value = value_
        self.dev = dev_

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
        Determine the last valid index by origin.

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
            self._devp = pd.Series(self.columns,name="devp")
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


    def to_tbl(self, drop_nas=True):
        """
        Transform triangle instance into a tabular representation.

        Parameters
        ----------
        drop_nas: bool
            Should records with NA values be dropped? Default value is True.

        Returns
        -------
        pd.DataFrame
        """
        tri = self.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        df = pd.melt(tri, id_vars=[self.origin], var_name=self.dev, value_name=self.value)
        if drop_nas:
            df = df[~np.isnan(df[self.value])]
        df = df.astype({self.origin:np.int_, self.dev:np.int_, self.value:np.float_})
        df = df[[self.origin, self.dev, self.value]].sort_values(by=[self.origin, self.dev])
        return(df.reset_index(drop=True))


    def __str__(self):
        formats_ = {dev_:"{:.0f}".format for dev_ in self.columns}
        return(self.to_string(formatters=formats_))


    def __repr__(self):
        formats_ = {dev_:"{:.0f}".format for dev_ in self.columns}
        return(self.to_string(formatters=formats_))




class _BaseIncrTriangle(_BaseTriangle):
    """
    Internal incremental triangle class definition.
    """
    def __init__(self, data, origin=None, dev=None, value=None):
        """
        Parameters
        ----------
        data: pd.DataFrame
            The dataset to be transformed into a triangle instance.
            ``data`` must be tabular loss data with at minimum columns
            representing the origin/acident year, development
            period and value of interest, given by ``origin``, ``dev``
            and ``value`` respectively.

        origin: str
            The fieldname in ``data`` representing origin year.

        dev: str
            The fieldname in ``data`` representing development period.

        value: str
            The fieldname in ``data`` representing loss amounts.
        """
        # Replace NaN values with 1.0 in value column.
        data.loc[np.where(np.isnan(data.value.values))[0], "value"] = 1.
        super().__init__(data, origin=origin, dev=dev, value=value)







class IncrTriangle(_BaseIncrTriangle):
    """
    Public incremental triangle class definition.
    """
    def __init__(self, data, origin=None, dev=None, value=None):
        """
        Parameters
        ----------
        data: pd.DataFrame
            The dataset to be transformed into a triangle instance.
            ``data`` must be tabular loss data with at minimum columns
            representing the origin/acident year, development
            period and value of interest, given by ``origin``, ``dev``
            and ``value`` respectively.

        origin: str
            The fieldname in ``data`` representing origin year.

        dev: str
            The fieldname in ``data`` representing development period.

        value: str
            The fieldname in ``data`` representing loss amounts.
        """
        super().__init__(data, origin=origin, dev=dev, value=value)


    def to_cum(self):
        """
        Transform triangle instance into cumulative representation.

        Returns
        -------
        trikit.triangle.CumTriangle
        """
        return(CumTriangle(self.to_tbl(), origin="origin", dev="dev", value="value"))





class _BaseCumTriangle(_BaseTriangle):
    """
    Internal cumulative triangle class definition.
    """
    def __init__(self, data, origin="origin", dev="dev", value="value"):
        """
        Transforms ``data`` into a cumulative triangle instance.

        Parameters
        ----------
         data: pd.DataFrame
            The dataset to be transformed into a triangle instance.
            ``data`` must be tabular loss data with at minimum columns
            representing the origin/acident year, development
            period and value of interest, given by ``origin``, ``dev``
            and ``value`` respectively.

        origin: str
            The fieldname in ``data`` representing the origin year.

        dev: str
            The fieldname in ``data`` representing the development period.

        value: str
            The fieldname in ``data`` representing loss amounts.
        """
        # Replace NaN values with 1.0 in value column.
        data2 = data.copy(deep=True)
        data2[value] = data2[value].map(lambda v: 1 if np.isnan(v) else v)
        data["cumval"] = data.groupby([origin], as_index=False)[value].cumsum()
        data = data.drop(value, axis=1)
        data = data.rename(columns={"cumval":value})
        super().__init__(data=data, origin=origin, dev=dev, value=value)

        # Properties.
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
            Not yet implemented.

        Returns
        -------
        float
        """
        arr = np.asarray(vals, dtype=np.float)
        return(np.NaN if arr.size==0 else stats.gmean(arr))


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
            Not yet implemented.

        Returns
        -------
        float
        """
        arr = np.asarray(vals, dtype=np.float)
        return(np.NaN if arr.size==0 else arr.mean())


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
        Update ``self.a2aind`` in order to down-weight ldfs in chain ladder
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

            In [1]: import trikit
            In [2]: raa = trikit.load(dataset="raa")
            In [3]: tri = trikit.totri(data=raa)
            In [4]: tri.a2a.iloc[:, :1]
            Out[1]:
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

            In [1]: tri.a2aind = (1982, 1, 0)
            In [2]: tri.a2aind
            Out[1]:
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

            In [1]: tri.a2aind = (1982, 1, 1)
            In [2]: tri.a2aind
            Out[1]:
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

            In [1]: tri.a2aind.at[1982, 1] = 0
            In [2]: tri.a2aind
            Out[1]:
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

        # Remove medial averages for the time being.
        self._a2a_avgs = self._a2a_avgs[~self._a2a_avgs.index.str.contains("medial")]

        return(self._a2a_avgs)



class CumTriangle(_BaseCumTriangle):
    """
    Public cumulative triangle class definition.
    """
    def __init__(self, data, origin=None, dev=None, value=None):

        super().__init__(data, origin=origin, dev=dev, value=value)


    def to_incr(self):
        """
        Obtain incremental triangle based on cumulative triangle instance.
        """
        incrtri = self.diff(axis=1)
        incrtri.iloc[:,0] = self.iloc[:, 0]
        incrtri = incrtri.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        df = pd.melt(incrtri, id_vars=["origin"], var_name="dev", value_name="value")
        df = df[~np.isnan(df["value"])].astype({"origin":np.int, "dev":np.int, "value":np.float})
        df = df.sort_values(by=["origin", "dev"]).reset_index(drop=True)
        return(IncrTriangle(df, origin="origin", dev="dev", value="value"))


    def plot(self, color="#334488", axes_style="darkgrid", context="notebook",
             col_wrap=4, **kwargs):
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
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_context(context)
        data = self.to_tbl()

        with sns.axes_style(axes_style):

            pltkwargs = dict(
                marker="o", markersize=7, alpha=1, markeredgecolor="#000000",
                markeredgewidth=.50, linestyle="--", linewidth=.75,
                fillstyle="full", color=color,
                )

            if kwargs:
                pltkwargs.update(kwargs)

            titlestr = "Loss Development by Origin"

            g = sns.FacetGrid(
                data, col="origin", col_wrap=col_wrap, margin_titles=False,
                despine=True, sharex=True, sharey=True,
                )

            g.map(plt.plot, "dev", "value", **pltkwargs)
            g.set_axis_labels("", "")
            g.set(xticks=data.dev.unique().tolist())
            g.set_titles("{col_name}", size=9)
            g.set_xticklabels(data.dev.unique().tolist(), size=8)

            for ii, _ in enumerate(g.axes):
                ax_ = g.axes[ii]

                ylabelss = [jj.get_text() for jj in list(ax_.get_yticklabels())]
                ylabelsn = [float(jj.replace(u"\u2212", "-")) for jj in ylabelss]
                ylabelsn = [jj for jj in ylabelsn if jj>=0]
                ylabels = ["{:,.0f}".format(jj) for jj in ylabelsn]
                if (len(ylabels)>0):
                    ax_.set_yticklabels(ylabels, size=8)
                ax_.tick_params(
                    axis='x', which='both', bottom=True, top=False, labelbottom=True
                    )

                # Draw border around each facet.
                for _, spine in ax_.spines.items():
                    spine.set_visible(True)
                    spine.set_color("#000000")
                    spine.set_linewidth(.50)


        # Adjust facets downward and and left-align figure title.
        # plt.subplots_adjust(top=0.9)
        # g.fig.suptitle(
        #     titlestr, x=0.065, y=.975, fontsize=11, color="#404040", ha="left"
        #     )
        plt.show()



    def cl(self, range_method=None, **kwargs):
        """
        Produce chain ladder estimates based on cumulative triangle instance.

        Parameters
        ----------
        range_method: {"bootstrap", "mack"}
            Specifies the method to use to quantify ultimate/reserve
            variability. When ``range_method=None``, reduces to the standard
            chain ladder technique providing reserve point estimates by
            origin. Defaults to None.

        kwargs: dict
            For each value of ``range_method``, there are a number of optional
            parameters that can be used to override the default behavior of
            the reserve estimator. If a keyword argument is provided that
            is not valid within the context of the given chain ladder variant,
            the argument will be ignored and a warning will be generated.
            What follows are valid optional keyword parameters for different
            values of ``range_method``:

            * ``range_method=None`` (standard chain ladder)
                - ``sel``: The ldf average to select from ``triangle.CumTriangle.a2a_avgs``.
                Defaults to ``"all-weighted"``.
                - ``tail``: Tail factor. Defaults to 1.0.

            * ``range_method="bootstrap"`` (bootstrap chain ladder)
                - ``sims``: The number of bootstrap resamplings to perform.
                Defaults to 1000.
                - ``q``: Determines which percentiles of the reserve distribution
                to compute. Defaults to [.75, .95].
                - ``neg_handler``: Dictates how negative triangle values should
                be handled. See documentation for ``_BoostrapChainLadder``
                for more information. Defaults to 1.
                - ``procdist``: The distribution used to incorporate process
                variance. At present , the only option is "gamma", but
                this wqill change in a future release.
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

        Returns
        -------
        chainladder.*ChainLadderResult
            One of {``BaseChainLadderResult``,``BootstrapChainLadderResult``,``MackChainLadderResult``}.

        Examples
        --------
        In the following examples we refer to the raa sample dataset. We read
        in the dataset and create a cumulative triangle instance, identified as
        ``tri``:

            In [1]: import trikit
            In [2]: raa = trikit.load("raa")
            In [3]: tri = trikit.totri(raa)


        1. Perform standard chain ladder, accepting defaults for ``sel`` and ``tail``:

            In [1]: cl0 = tri.cl()
            In [2]:
            Out[1]:

        2. Perform standard chain ladder, updating values for ``sel`` and ``tail``:

            In [1]: kwds = dict(sel="medial-5", tail=1.015)
            In [2]: cl1 = tri.cl(**kwds)
            In [3]:
            Out[1]:


        3. Perform boostrap chain ladder, overriding ``sims``, ``q`` and ``symmetric``:

            In [1]: kwds = dict(sims=2500, q=[.90, .99], symmetric=True)
            In [2]: bcl = tri.cl(range_method="bootstrap", **kwds)
            In [3]:
            Out[1]:


        4. Perfrom Mack chain ladder, overriding ``alpha``:

            In [1]: kwds = {"alpha":2}
            In [2]: mcl = tri.cl(range_method="mack", **kwds)
            In [3]:
            Out[1]:

        """
        kwds = {} if kwargs is None else kwargs

        if range_method is None:
            result = BaseChainLadder(self).__call__(**kwds)

        elif range_method.lower().strip().startswith("boot"):
            result = BootstrapChainLadder(self).__call__(**kwds)

        elif range_method.lower().strip().startswith("mack"):
            result = MackChainLadder(self).__call__(**kwds)

        elif range_method.lower().startswith("mcmc"):
            raise NotImplementedError("range_method='mcmc' not yet implemented.")

        else:
            raise ValueError("Invalid range_method specification: {}".format(range_method))

        return(result)






def totri(data, type_="cum", data_format="incr", data_shape="tabular",
          origin="origin", dev="dev", value="value"):
    """
    Create a triangle object based on ``data``. ``type_`` can be one of
    "incr" or "cum", determining whether the resulting triangle represents
    incremental or cumulative losses/counts/alae.
    If ``data_shape="triangle"``, ``data`` is assumed to be structured as a
    runoff triangle, indexed by origin with columns representing development
    periods. If ``data_shape="tabular"``, data is assumed to be tabular with at
    minimum columns ``origin``, ``dev`` and ``value``, which represent origin
    year, development period and metric of interest respectively.
    ``data_format`` indicates whether the metric of interest are cumulative
    or incremental in nature. Default value is "incr".

    Parameters
    ----------
    data: pd.DataFrame
        The dataset to be coerced into a triangle instance. ``data`` can be
        tabular loss data, or a dataset (pandas DataFrame) formatted as a
        triangle, but not typed as such. In the latter case,
        ``data_shape`` should be set to ```triangle``.

    type_: {"cum", "incr"}
        Either "cum" or "incr". Specifies how the metric of interest (losses,
        counts, alae, etc.) are to be represented in the returned triangle
        instance.
        ``type_`` can also be specified as "i" for "incremental" or "c" for
        "cumulative". Default value is "cum".

    data_format: {"cum", "incr"}
        Specifies the representation of the metric of interest in ``data``.
        Default value is "incr".

    data_shape:{"tabular", "triangle")
        Indicates whether ``data`` is formatted as a triangle as opposed
        to tabular loss data. In some workflows, triangles may have already
        been created, and are available in auxillary files. In such cases, the
        triangle formatted data will be passed in as a DataFrame, and
        converted into the desired representation directly. Default value is
        False.

    origin: str
        The field in ``data`` representing the origin year. When
        ``has_tri_shape`` is False, ``origin`` is ignored. Default value is
        "origin".

    dev: str
        The field in ``data`` representing the development period. When
        ``has_tri_shape`` is False, ``dev`` is ignored. Default value is
         "dev".

    value: str
        The field in ``data`` representing loss amounts. When
        ``has_tri_shape`` is False, ``value`` is ignored. Default value is
        "value".

    Returns
    -------
    {trikit.triangle.IncrTriangle, trikit.triangle.CumTriangle}
    """
    if data_shape=="triangle":

        if data_format.lower().strip().startswith("i"):
            # data is in incremental triangle format (but not typed as such).
            inctri = data.reset_index(drop=False).rename({"index":"origin"}, axis=1)
            df = pd.melt(inctri, id_vars=["origin"], var_name="dev", value_name="value")

        elif data_format.lower().strip().startswith("c"):
            # data is in cumulative triangle format (but not typed as such).
            incrtri = data.diff(axis=1)
            incrtri.iloc[:,0] = data.iloc[:, 0]
            incrtri = incrtri.reset_index(drop=False).rename({"index":"origin"}, axis=1)
            df = pd.melt(incrtri, id_vars=["origin"], var_name="dev", value_name="value")
            df = df[~np.isnan(df["value"])].astype({"origin":np.int, "dev":np.int, "value":np.float})

        else:
            raise NameError("Invalid data_format argument: `{}`.".format(type_))

        df = df[~np.isnan(df["value"])].astype({"origin":np.int, "dev":np.int, "value":np.float})
        df = df.sort_values(by=["origin", "dev"]).reset_index(drop=True)

    elif data_shape=="tabular":

        if data_format.lower().strip().startswith("c"):
            df = data.rename({value:"cum"}, axis=1)
            df["incr"] = df.groupby([origin])["cum"].diff(periods=1)
            df["incr"] = np.where(np.isnan(df["incr"]), df["cum"], df["incr"])
            df = df.drop("cum", axis=1).rename({"incr":value}, axis=1)

        else:
            df = data

    else:
        raise NameError("Invalid data_shape argument: `{}`.".format(data_shape))

    df = df.reset_index(drop=True)

    # Transform df to triangle instance.
    if type_.lower().startswith("i"):
        tri = IncrTriangle(data=df, origin=origin, dev=dev, value=value)

    elif type_.lower().startswith("c"):
        tri = CumTriangle(data=df, origin=origin, dev=dev, value=value)

    return(tri)
