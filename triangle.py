"""
This module contains the definitions of both the ``_IncrTriangle`` and
``_CumTriangle`` classes. Users should avoid instantiating ``_IncrTriangle``
or ``_CumTriangle`` instances directly; rather the dataset and triangle
arguments should be passed to ``totri``, which will return either an
instance of ``_CumTriangle`` or ``_IncrTriangle``, depending on the argument
specified for ``type_``.
"""
import itertools
import numpy as np
import pandas as pd


class _IncrTriangle(pd.DataFrame):
    """
    ``_IncrTriangle`` specification. Note that this class isn't part of
    trikit's public interface, as is used  primarily as a base class for
    ``_CumTriangle`` object definitions. As such, users shouldn't
    instantiate ``_IncrTriangle`` instances directly. To obtain an
    incremental triangle object, use ``utils.totri``.
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
            The field in ``data`` representing the origin year. When
            ``trifmt`` is not None, ``origin`` is ignored. Defaults to None.

        dev: str
            The field in ``data`` representing the development period. When
            ``trifmt`` is not None, ``dev`` is ignored. Defaults to None.

        value: str
            The field in ``data`` representing loss amounts. When ``trifmt``
            is not None, ``value`` is ignored. Defaults to None.

        Returns
        -------
        pd.DataFrame
        """
        try:
            if all(i is None for i in (origin, dev, value)):
                origin, dev, value = "origin", "dev", "value"
            dat_init = data[[origin, dev, value]]
            dat_init = dat_init.groupby([origin, dev], as_index=False).sum()
            tri = dat_init.pivot(index=origin, columns=dev).rename_axis(None)
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
        self._maturity = None
        self._triind = None
        self._devp = None
        self._latest = None
        self._origins = None
        self._rlvi = None
        self._clvi = None



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
        Determine the value on the triangle's latest diagonal. Loss amounts
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
        Determine the latest loss amounts by origin year.

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
        Determine the latest loss amounts by development period.

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
        Determine triangle's development periods.

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
        Determine triangle's origin periods.

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
        Determine maturity for each origin period.

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




class _CumTriangle(_IncrTriangle):
    """
    Cumulative triangle class definition.
    """
    def __init__(self, data, origin=None, dev=None, value=None):
        """
        Attempts to represent ``data`` as a cumulative triangle instance.
        ``origin``, ``dev`` and ``value`` arguments represent fieldnames
        corresponding to loss year, development period and loss amount
        respectively.

        Parameters
        ----------
        origin: str
            The field in ``data`` representing the origin year.

        dev: str
            The field in ``data`` representing the development period.

        value: str
            The field in ``data`` representing loss amount.

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
        Compute the geometric average of the elements of vals.

        Parameters
        ----------
        vals: np.ndarray
            An array of values, typically representing link ratios from a
            single development period.

        weights: np.ndarray
            Weights to assign specific values in the average computation.
            If None, each value is given equal weight.

        Returns
        -------
        float
        """
        if len(vals)==0:
            return(None)
        vals = list(vals)
        return(np.prod(vals) ** (1 / len(vals)))


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
            If None, each value is given equal weight.

        Returns
        -------
        float
        """
        if len(vals) == 0: return (None)
        return(sum(i for i in vals) / len(vals))


    @staticmethod
    def _medial(vals, weights=None):
        """
        Compute the medial average of elements in vals. Medial average
        eliminates the min and max values, then returns the arithmetic
        average of the remaining items.

        Parameters
        ----------
        vals: np.ndarray
            An array of values, typically representing link ratios from a
            single development period.

        weights: np.ndarray
            Weights to assign specific values in the average computation.
            If None, each value is given equal weight.

        Returns
        -------
        float
        """
        vals = list(vals)
        if len(vals) == 0: avg = None
        if len(vals) == 1:
            avg = vals[0]
        elif len(vals) == 2:
            avg = sum(vals) / len(vals)
        else:
            max_indx = vals.index(max(vals))
            vals.remove(vals[max_indx])
            min_indx = vals.index(min(vals))
            vals.remove(vals[min_indx])
            avg = sum(vals)/len(vals)
        return(avg)


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
        Update self.a2aind for use in calculating Mack ldfs.

        Parameters
        ----------
        update_spec: tuple
            3-tuple consisting of ``(index, column, value)``, representing
            the intersection point of the ``self.a2a`` target cell, and the
            value used to update it.

        Examples
        --------
        Load raa sample dataset, and remove a highly-leveraged age-to-age
        factor from influencing the LDF calculation.

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
        """
        indx, column, value = update_spec
        self._a2aind.at[indx, column] = value


    @property
    def a2a_avgs(self):
        """
        Compute age-to-age factors based on self.a2a triangle of adjacent
        proportions. Averages computed include "simple", "geometric", "medial",
        "weighted" and "mack", where "mack" represent LDFs calculated in
        accordance with "Mack's alpha", set to 0, 1 or 2.

        Returns
        -------
        pd.DataFrame
        """
        if self._a2a_avgs is None:

            _nbr_periods = list(range(1, self.a2a.shape[0]))
            _nbr_periods.append(0)
            indxstrs = list()

            # Create lookup table for average functions.
            avgfuncs = {
                'simple'   :self._simple,
                'geometric':self._geometric,
                'medial'   :self._medial,
                'weighted' :None
                }

            # Remove `0` entry, and add as last element of list.
            ldf_avg_lst = list(
                itertools.product(avgfuncs.keys(), _nbr_periods)
                )

            for i in ldf_avg_lst:
                iteravg, iterdur = i[0], i[1]
                iterstr = "all-" + str(iteravg) if iterdur==0 \
                              else str(iterdur) + "-" + str(iteravg)
                indxstrs.append(iterstr)

            indx = sorted(ldf_avg_lst, key=lambda x: x[1])

            self._a2a_avgs = \
                pd.DataFrame(index=indxstrs, columns=self.a2a.columns)

            for a in enumerate(ldf_avg_lst):
                duration, avgtype, indxpos = a[1][1], a[1][0], a[0]
                indxstr, iterfunc = indxstrs[indxpos], avgfuncs[avgtype]
                for col in range(self.a2a.shape[1]):
                    itercol, colstr = self.a2a.iloc[:, col], self.a2a.columns[col]

                    if avgtype=='weighted':
                        t_ic_1, t_ic_2 = self.iloc[:, col], self.iloc[:, (col + 1)]
                        # Find first NaN value in t_ic_2.
                        first_nan_year  = t_ic_2.index[t_ic_2.count():][0]
                        first_nan_indx  = t_ic_2.index.searchsorted(first_nan_year)
                        final_cell_indx = first_nan_indx
                        if duration==0:
                            first_cell_indx = 0
                        else:
                            first_cell_indx = (final_cell_indx-duration) if \
                                              (final_cell_indx-duration)>=0 else 0

                        # Divide sum of t_ic_2 by t_ic_1.
                        ic_2     = t_ic_2[first_cell_indx:final_cell_indx]
                        ic_1     = t_ic_1[first_cell_indx:final_cell_indx]
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


    def plot_devp(self, facets=False, file=None, **kwargs):
        """
        Visualize triangle development patterns. If file is given, save plot
        to that location.

        Parameters
        ----------
        cumtri: cumulative._CumTriangle
            An instance of cumulative._CumTriangle.

        facets: bool
            If True, loss development plots for each origin year are faceted,
            otherwise development patterns are plotted together on a single set
            of axes.

        file: str
            Location to save generated plot. Defualts value is None.

        Returns
        -------
        matplotlib.pyplot.plot
            Rendering of each origin year's cumulative losses as a function of
            development period.
        """
        context    = kwargs.get("context", "notebook")
        axes_style = kwargs.get("axes_style", "darkgrid")
        palette    = kwargs.get("palette", "Accent")
        dat        = _tritotbl(self)

        sns.set_context(context) # talk poster paper notebook

        if facets:

            with sns.axes_style(axes_style):
                g = sns.FacetGrid(
                    dat,col="origin",col_wrap=5,margin_titles=False,
                    ylim=(0, dat.value.max())
                    )
            g.map(plt.plot,"dev","value", color="#334488",alpha=.7)
            g.map(plt.scatter,"dev","value", s=20,color="#334488",alpha=.7)
            g.set_axis_labels("dev_period", "")
            g.set(xticks=dat.dev.unique().tolist())
            g.fig.subplots_adjust(wspace=.025)

        else:

            with sns.axes_style(axes_style,{'legend.frameon':True}):
                g = sns.pointplot(
                    x="dev", y="value", hue="origin",
                    data=dat, palette=sns.color_palette(palette, dat.dev.unique().size),
                    legend=False
                    )
            g.set_title("Loss Development by Origin Year", loc="left")
            legend = plt.legend(frameon=1, loc='lower right')
            frame = legend.get_frame()
            frame.set_color('white')

        # fig = g.get_figure()
        # fig.savefig("devps.pdf") # this does not work in Seaborn 0.7.1

        # matplotlib way:
        # plt.savefig('yourTitle.png')

        # ax = sns.violinplot(x="Gender", y="Salary", hue="Degree", data=job_data)
        # #Returns the :class:~matplotlib.figure.Figure instance the artist belongs to
        # fig = ax.get_figure()
        # fig.savefig('gender_salary.png')

        # fig = plt.Figure(). Then you can save the figure with fig.savefig()
        return(None)


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




























