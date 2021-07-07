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
from .estimators.chainladder import BaseChainLadder
from .estimators.chainladder.bootstrap import BootstrapChainLadder
from .estimators.chainladder.mack import MackChainLadder
from .estimators.glm import GLM




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
            The fieldname in ``data`` representing origin period.

        dev: str
            The fieldname in ``data`` representing development period.

        value: str
            The fieldname in ``data`` representing loss amounts.
        """
        self._validate(data, origin=origin, dev=dev, value=value)
        origin_ = "origin" if origin is None else origin
        dev_ = "dev" if dev is None else dev
        value_ = "value" if value is None else value

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




    @staticmethod
    def _validate(data, origin=None, dev=None, value=None):
        """
        Ensure data has requisite columns.

        Parameters
        ----------
        data: pd.DataFrame
            Initial dataset to be coerced to triangle.

        origin: str
            The fieldname in ``data`` representing origin period.

        dev: str
            The fieldname in ``data`` representing development period.

        value: str
            The fieldname in ``data`` representing loss amounts.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be an instance of pd.DataFrame.")

        origin_ = "origin" if origin is None else origin
        if origin_ not in data.columns:
            raise AttributeError("`{}` not present in data.".format(origin_))

        dev_ = "dev" if dev is None else dev
        if dev_ not in data.columns:
            raise AttributeError("`{}` not present in data.".format(dev_))

        value_ = "value" if value is None else value
        if value_ not in data.columns:
            raise AttributeError("`{}` not present in data.".format(value_))


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
            self._triind = pd.DataFrame(columns=self.columns, index=self.index)
            self._triind.iloc[:,:] = 0
            for devp in self.clvi.index:
                last_actual_origin = self.clvi[self.clvi.index==devp].origin.values[0]
                last_actual_offset = self.clvi[self.clvi.origin==last_actual_origin].row_offset.values[0]
                self._triind.iloc[(last_actual_offset + 1):,self.columns.get_loc(devp)] = 1
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
            lindx = self.apply(lambda devp: devp.last_valid_index(), axis=1)
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



    def diagonal(self, offset=0):
        """
        Return triangle values at given offset. When ``offset=0``, returns
        latest diagonal.

        Parameters
        ----------
        offset: int
            Negative integer value (or 0) representing the diagonal to return.
            To return the second diagonal, set ``offset=-1``. If abs(offset)
            exceeds (number of development periods - 1), ``ValueError`` is raised.
            Default value is 0 (represents latest diagonal).

        Returns
        -------
        pd.Series
        """
        if np.abs(offset)>(self.devp.size - 1):
            raise ValueError("abs(offset) cannot exceed the number of development periods.")
        df = self.latest.copy()
        df["latest"] = np.NaN
        df = df.reset_index(drop=False).rename(
            {"index":"origin_indx"}, axis=1)[["origin_indx"]]
        df["dev_indx"] = df["origin_indx"].values[::-1]
        df["dev_indx"] = df["dev_indx"] + offset
        df = df[df.dev_indx>=0].reset_index(drop=True)
        df = df.assign(
            origin=df["origin_indx"].map(lambda v: self.origins[v]),
            dev=df["dev_indx"].map(lambda v: self.devp[v]),
            value=df.apply(lambda rec: self.iat[rec.origin_indx, rec.dev_indx], axis=1)
            )
        return(df[["origin", "dev", "value"]])







    def to_tbl(self, dropna=True):
        """
        Transform triangle instance into a tabular representation.

        Parameters
        ----------
        dropna: bool
            Should records with NA values be dropped? Default value is True.

        Returns
        -------
        pd.DataFrame
        """
        tri = self.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        df = pd.melt(tri, id_vars=[self.origin], var_name=self.dev, value_name=self.value)
        if dropna:
            df = df[~np.isnan(df[self.value])]
        df = df.astype({self.origin:np.int_, self.dev:np.int_, self.value:np.float_})
        df = df[[self.origin, self.dev, self.value]].sort_values(by=[self.origin, self.dev])
        return(df.reset_index(drop=True))


    def __str__(self):
        formats = {devp:"{:,.0f}".format for devp in self.columns}
        return(self.to_string(formatters=formats))


    def __repr__(self):
        formats = {devp:"{:,.0f}".format for devp in self.columns}
        return(self.to_string(formatters=formats))




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
        # data.loc[np.where(np.isnan(data.value.values))[0], "value"] = 1.
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
            period and incremental value of interest, given by ``origin``,
            ``dev`` and ``value`` respectively.

        origin: str
            The fieldname in ``data`` representing the origin year.

        dev: str
            The fieldname in ``data`` representing the development period.

        value: str
            The fieldname in ``data`` representing incremental loss amounts.
        """
        # Replace NaN values with 1.0 in value column.
        data2 = data.copy(deep=True)
        # data2[value] = data2[value].map(lambda v: 1 if np.isnan(v) else v)
        data["cumval"] = data.groupby([origin], as_index=False)[value].cumsum()
        data = data.drop(value, axis=1)
        data = data.rename(columns={"cumval":value})
        super().__init__(data=data, origin=origin, dev=dev, value=value)

        # Properties.
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
            value used to update it. ``value`` must be either 0 or 1.

        Examples
        --------
        Load raa sample dataset, and remove a highly-leveraged age-to-age
        factor from influencing the ldf calculation::

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
        period 1, run the following::

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
        to 1, simply run::

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
        methods directly::

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
        # vals = list(vals)
        # if len(vals)==0:
        #     avg_ = None
        # elif len(vals)==1:
        #     avg_ = vals[0]
        # elif len(vals)==2:
        #     avg_ = sum(vals) / len(vals)
        # else:
        #     keep_ = sorted(vals)[1:-1]
        #     avg_ = sum(keep_) / len(keep_)
        # return(avg_)

        if weights is None:
            w = np.ones(len(vals))
        else:
            w = weights
            if len(w)!=len(vals):
                raise ValueError("`vals` and `weights` must have same size")

        # Return first element of arr_all if all array elements are the same.
        arr_all = np.sort(np.asarray(vals))
        if np.all(arr_all==arr_all[0]):
            avg = arr_all[0]

        elif arr_all.shape[0]==1:
            avg = arr_all[0]

        elif arr_all.shape[0]==2:
            avg = (w * arr_all).sum() / w.sum()

        else:
            medial_indicies = np.where(np.logical_and(arr_all!=arr_all.min(), arr_all!=arr_all.max()))
            arr = arr_all[medial_indicies]
            w = w[medial_indicies]

            if arr.shape[0]==0:
                avg = np.NaN
            else:
                avg = (w * arr).sum() / w.sum()

        return(avg)


    def a2a_avgs(self):
        """
        Compute age-to-age factors based on ``self.a2a`` table of adjacent
        proportions. Averages computed include "simple", "geometric", "medial"
        and "weighted".

        Returns
        -------
        pd.DataFrame
        """
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
            "all-" + str(ii[0]) if ii[1]==0 else "{}-{}".format(ii[0], ii[1])
                for ii in ldf_avg_lst
            ]

        indx = sorted(ldf_avg_lst, key=lambda v: v[1])
        _a2a_avgs = pd.DataFrame(index=indxstrs, columns=self.a2a.columns)
        a2a_adj = self.a2a * self.a2aind

        for a in enumerate(ldf_avg_lst):
            duration, avgtype, indxpos = a[1][1], a[1][0], a[0]
            indxstr, iterfunc = indxstrs[indxpos], avgfuncs[avgtype]

            for col in range(a2a_adj.shape[1]):
                itercol, colstr = a2a_adj.iloc[:, col], a2a_adj.columns[col]

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
                        iteravg = (sum_ic_2 / sum_ic_1)
                    except ZeroDivisionError:
                        iteravg = np.NaN

                else: # avgtype in ('simple', 'geometric', 'medial')
                    # Find index of first row with NaN.
                    if any(itercol.map(lambda x: np.isnan(x))):
                        first_nan_year = itercol.index[itercol.apply(lambda x: np.isnan(x))][0]
                        first_nan_indx = itercol.index.searchsorted(first_nan_year)
                        final_cell_indx = first_nan_indx
                        if duration==0:
                            first_cell_indx = 0
                        else:
                            first_cell_indx = (final_cell_indx - duration) if \
                                              (final_cell_indx - duration)>=0 else 0

                    else: # itercol has 0 NaN's
                        final_cell_indx = len(itercol)
                        first_cell_indx = 0 if duration==0 else (final_cell_indx-duration)
                    try:
                        link_ratios = itercol[first_cell_indx:final_cell_indx]
                        iteravg = iterfunc(link_ratios[link_ratios>0])
                    except ZeroDivisionError:
                        iteravg = np.NaN

                _a2a_avgs.loc[indxstr, colstr] = iteravg

        # Remove medial averages for the time being.
        #_a2a_avgs = _a2a_avgs[~_a2a_avgs.index.str.contains("medial")]

        return(_a2a_avgs)



class CumTriangle(_BaseCumTriangle):
    """
    Cumulative triangle class definition.
    """
    def __init__(self, data, origin=None, dev=None, value=None):

        super().__init__(data, origin=origin, dev=dev, value=value)


    def to_incr(self):
        """
        Obtain incremental triangle based on cumulative triangle instance.

        Returns
        -------
        trikit.triangle.IncrTriangle

        Examples
        --------
        Convert existing cumulative triangle instance into an instance of
        ``trikit.triangle.IncrTriangle``::

            In [1]: from trikit import load, totri
            In [2]: cumtri = totri(load("raa"))
            In [3]: incrtri = cumtri.to_incr()
            In [4]: type(incrtri)
            Out[1]: triangle.IncrTriangle
        """
        incrtri = self.diff(axis=1)
        incrtri.iloc[:,0] = self.iloc[:, 0]
        incrtri = incrtri.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        df = pd.melt(incrtri, id_vars=["origin"], var_name="dev", value_name="value")
        df = df[~np.isnan(df["value"])].astype({"origin":np.int, "dev":np.int, "value":np.float})
        df = df.sort_values(by=["origin", "dev"]).reset_index(drop=True)
        return(IncrTriangle(df, origin="origin", dev="dev", value="value"))



    def plot(view="combined", **kwargs):
        """
        Plot cumulative loss development over a single set of axes or
        as faceted-by-origin exhibit.

        Parameters
        ----------
        view: {"combined", "faceted"}
            Whether to display cumulative loss development in a single or faceted view.
            Default value is ``"combined"``.

        kwargs: dict

            Options for combined view:

                cmap: str
                    Selected matplotlib color map. For additional options, visit:
                    https://matplotlib.org/tutorials/colors/colormaps.html.

            Options for faceted view:

                color: str
                    Color to plot loss development in each facet. Default value is "#334488".

                axes_style: str
                    Aesthetic style of plots. Defaults to "darkgrid". Other options
                    include: {whitegrid, dark, white, ticks}.

                context: str
                    Set the plotting context parameters. According to the seaborn
                    documentation, This affects things like the size of the labels,
                    lines, and other elements of the plot, but not the overall style.
                    Defaults to ``"notebook"``. Additional options include
                    {"paper", "talk", "poster"}.
        """
        if view.startswith("f"):
            self._faceted_view(**kwargs)
        else:
            self._combined_view(**kwargs)


    def _combined_view(**kwargs):
        """
        Visualize triangle loss development using a combined view.

        Parameters
        ----------
        cmap: str
            Selected matplotlib color map. For additional options, visit:
            https://matplotlib.org/tutorials/colors/colormaps.html.

        kwargs: dict
            Additional plot styling options.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        pltkwargs = dict(
            marker="s", markersize=5, alpha=1, linestyle="-", linewidth=1.5,
            figsize=(9, 6), cmap="hsv",
            )

        if kwargs:
            pltkwargs.update(kwargs)

        data = self.to_tbl()
        grps = data.groupby("origin", as_index=False)
        data_list = [grps.get_group(ii) for ii in self.origins]
        xticks = np.sort(data.dev.unique())

        # Get unique hex color for each unique origin period.
        fcolors = cm.get_cmap(pltkwargs["cmap"], len(self.origins))
        colors_rgba = [fcolors(ii) for ii in np.linspace(0, 1, len(self.origins))]
        colors_hex = [mpl.colors.to_hex(ii, keep_alpha=False) for ii in colors_rgba]

        fig, ax = plt.subplots(1, 1, figsize=pltkwargs["figsize"], tight_layout=True)

        ax.set_title("Loss Development by Origin", fontsize=9, loc="left")

        for hex_color, dforg in zip(colors_hex, data_list):

            ax.plot(
                dforg.dev.values, dforg.value.values / 1000, color=hex_color,
                linewidth=pltkwargs["linewidth"], linestyle=pltkwargs["linestyle"],
                label=dforg.origin.values[0], marker=pltkwargs["marker"],
                markersize=pltkwargs["markersize"]
                )

        # Reduce thickness of plot outline.
        for axis in ["top","bottom","left","right"]:
            ax.spines[axis].set_linewidth(0.5)

        ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: format(int(v), ",")))
        ax.set_xlabel("dev", fontsize=8)
        ax.set_ylabel("(000's)", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", which="major", direction="in", labelsize=8)
        ax.tick_params(axis="y", which="major", direction="in", labelsize=8)
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        ax.grid(True)
        ax.legend(loc="lower right", fancybox=True, framealpha=1, fontsize="x-small")

        plt.show()


    def _faceted_view(self, color="#334488", axes_style="darkgrid", context="notebook",
                      col_wrap=4, **kwargs):
        """
        Visualize triangle loss development using a faceted view.

        Parameters
        ----------
        color: str
            Color to plot loss development in each facet. Default value is "#334488".

        axes_style: str
            Aesthetic style of plots. Defaults to "darkgrid". Other options
            include: {whitegrid, dark, white, ticks}.

        context: str
            Set the plotting context parameters. According to the seaborn
            documentation, This affects things like the size of the labels,
            lines, and other elements of the plot, but not the overall style.
            Defaults to ``"notebook"``. Additional options include
            {"paper", "talk", "poster"}.

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

            g = sns.FacetGrid(
                data, col="origin", col_wrap=col_wrap, margin_titles=False,
                despine=True, sharex=True, sharey=True,
                )

            g.map(plt.plot, "dev", "value", **pltkwargs)
            g.set_axis_labels("", "")
            g.set_titles("{col_name}", size=9)
            g.set(xticks=data.dev.unique().tolist())
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

        plt.show()


    def base_cl(self, sel="all-weighted", tail=1.0):
        """
        Produce chain ladder reserve estimates based on cumulative triangle instance.

        Parameters
        ----------
       sel: str, pd.Series or array_like
            If ``sel`` is a string, the specified loss development patterns will be
            the associated entry from ``self.tri.a2a_avgs``.
            If ``sel`` is array_like, values will be used in place of loss development
            factors computed from the traingle directly. For a triangle with n development
            periods, ``sel`` should be array_like with length n - 1.
            Defaults to "all-weighted".

        tail: float
            Chain ladder tail factor. Defaults to 1.0.

        custom_ldfs: array_like
            An alternative set of loss development factors derived outside
            of available patterns in ``triangle.CumTriangle.a2a_avgs``.
            If ``custom_ldfs`` is specified, ``sel`` is ignored. ``custom_ldfs``
            should be array_like with length (tri.devp.size - 1). Defaults to None.


        Examples
        --------
        Generate chain ladder reserve point estimates using the raa dataset.
        ``tri`` is first created using the raa dataset::

            In [1]: import trikit
            In [2]: raa = trikit.load("raa")
            In [3]: tri = trikit.totri(raa)
            In [4]: cl = tri.base_cl()


        Perform standard chain ladder, updating values for ``sel`` and ``tail``::

            In [5]: cl = tri.base_cl(sel="medial-5", tail=1.015)


        Provide custom sequence of loss development factors::

            In [6]: ldfs = [5., 2.5, 1.25, 1.15, 1.10, 1.05, 1.025, 1.01, 1.005,]
            In [7]: cl = tri.base_cl(sel=ldfs, tail=1.001)
        """
        kwds = dict(sel=sel, tail=tail)
        return(BaseChainLadder(self).__call__(**kwds))


    def boot_cl(self, sims=1000, q=[.75, .95], procdist="gamma", parametric=False,
                two_sided=False, interpolation="linear", random_state=None):
        """
        Estimate reserves and the distribution of reserve outcomes by origin and in
        total via bootstrap resampling. The estimated distribution of losses assumes
        development is completen by the final development period in order to avoid the
        complication of modeling a tail factor.

        Parameters
        ----------
        sims: int
            The number of bootstrap simulations to perform. Defaults to 1000.

        q: array_like of float or float
            Quantile or sequence of quantiles to compute, which must be
            between 0 and 1 inclusive.

        procdist: str
            The distribution used to incorporate process variance. Currently,
            this can only be set to "gamma".

        two_sided: bool
            Whether to include the two_sided interval in summary output. For example,
            if ``two_sided==True`` and ``q=.95``, the 2.5th and 97.5th quantiles of the
            bootstrapped reserve  distribution will be returned [(1 - .95) / 2, (1 + .95) / 2].
            When False, only the specified quantile(s) will be computed. Defaults
            to False.

        parametric: bool
            If True, fit standardized residuals to a normal distribution via maximum likelihood,
            and sample from the resulting distribution. Otherwise, values are sampled with
            replacement from the collection of standardized residuals. Defaults to False.

        interpolation: {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use
            when the desired quantile lies between two data points i < j:

                - linear: i + (j - i) * fraction, where fraction is the fractional
                part of the index surrounded by i and j.
                - lower: i.
                - higher: j.
                - nearest: i or j, whichever is nearest.
                - midpoint: (i + j) / 2.

        random_state: np.random.RandomState
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        BootstrapChainLadderResult


        Examples
        --------
        Generate boostrap chain ladder reserve estimates. ``tri`` is first created
        using the raa dataset::

            In [1]: import trikit
            In [2]: raa = trikit.load("raa")
            In [3]: tri = trikit.totri(raa)
            In [4]: bcl = tri.boot_cl()
        """
        kwds =  dict(
            sims=sims, q=q, procdist=procdist, parametric=parametric, two_sided=two_sided,
            interpolation=interpolation, random_state=random_state
            )
        return(BootstrapChainLadder(self).__call__(**kwds))


    def mack_cl(self, alpha=1, tail=1.0, dist="lognorm", q=[.75, .95], two_sided=False):
        """
        Return a summary of ultimate and reserve estimates resulting from
        the application of the development technique over self.tri. Summary
        DataFrame is comprised of origin year, maturity of origin year, loss
        amount at latest evaluation, cumulative loss development factors,
        projected ultimates and the reserve estimate, by origin year and in
        aggregate.

        ### TODO ###
        Allow for tail factor other than 1.0.

        Parameters
        ----------
        alpha: {0, 1, 2}
            * ``0``: Straight average of observed individual link ratios.
            * ``1``: Historical Chain Ladder age-to-age factors.
            * ``2``: Regression of $C_{k+1}$ on $C_{k}$ with 0 intercept.

        tail: float
            Tail factor. Currently not implemented. Will be available in
            a future release.

        dist: {"norm", "lognorm"}
            The distribution function chosen to approximate the true
            distribution of reserves by origin period. In Mack[1], if the
            volume of outstanding claims is large enough, due to the central
            limit theorem, we can assume that the distribution function is
            Normal with expected value equal to the point estimate given by
            $R_{i}$ and standard deviation equal to the standard error of
            $R_{i}$, $s.e.(R_{i})$. It is also noted that if the true
            distribution of reserves is skewed, the Normal may not serve as a
            good approximation, and it may be preferrable to opt for
            the Log-normal distribution.

            * If ``dist="norm"``, the Normal distribution will be used to
            estimate reserve quantiles.
            * If ``dist="lognorm"``, the Log-normal distribution will be used
            to estimate reserve quantiles.

        q: array_like of float
            Quantile or sequence of quantiles to compute, which must be
            between 0 and 1 inclusive.

        two_sided: bool
            Whether the two_sided interval should be included in summary
            output. For example, if ``two_sided==True`` and ``q=.95``, then
            the 2.5th and 97.5th quantiles of the estimated reserve
            distribution will be returned [(1 - .95) / 2, (1 + .95) / 2]. When
            False, only the specified quantile(s) will be computed. Defaults
            to False.

        Returns
        -------
        MackChainLadderResult


        Examples
        --------
        Generate Mack chain ladder reserve estimates. ``tri`` is first created
        using the raa dataset. In the call to ``mack_cl``, ``alpha`` is set to
        2, and ``two_sided=True``::

            In [1]: import trikit
            In [2]: raa = trikit.load("raa")
            In [3]: tri = trikit.totri(raa)
            In [4]: mcl = tri.mack_cl(alpha=2, two_sided=True)
        """
        kwds = dict(alpha=alpha, tail=tail, dist=dist, q=q, two_sided=two_sided)
        return(MackChainLadder(self).__call__(**kwds))


    def mcmc_cl(self):
        """
        Markov Chain Monte Carlo reserve estimator.
        """
        raise NotImplementedError("mcmc_cl not yet implemented.")

    def glm_estimator(self, var_power=2):
        """
        Generate reserve estimates via Generalized Linear Model framework.
        Note that ``glm_estimator`` assumes development is complete by the final
        development period. GLMs are fit using statsmodels Tweedie family
        """



def totri(data, tri_type="cum", data_format="incr", data_shape="tabular",
          origin="origin", dev="dev", value="value"):
    """
    Create a triangle object based on ``data``. ``tri_type`` can be one of
    "incr" or "cum", determining whether the resulting triangle represents
    incremental or cumulative losses/counts.
    If ``data_shape="triangle"``, ``data`` is assumed to be structured as a
    runoff triangle, indexed by origin with columns representing development
    periods. If ``data_shape="tabular"``, data is assumed to be tabular with at
    minimum columns ``origin``, ``dev`` and ``value``, which represent origin
    year, development period and metric of interest respectively.
    ``data_format`` specifies whether the metric of interest are cumulative
    or incremental in nature. Default value is "incr".

    Parameters
    ----------
    data: pd.DataFrame
        The dataset to be coerced into a triangle instance. ``data`` can be
        tabular loss data, or a dataset (pandas DataFrame) formatted as a
        triangle, but not typed as such. In the latter case,
        ``data_shape`` should be set to "triangle".

    tri_type: {"cum", "incr"}
        Either "cum" or "incr". Specifies how the measure of interest (losses,
        counts, alae, etc.) should be represented in the returned triangle
        instance.

    data_format: {"cum", "incr"}
        Specifies the representation of the metric of interest in ``data``.
        Default value is "incr".

    data_shape:{"tabular", "triangle"}
        Indicates whether ``data`` is formatted as a triangle instead of
        tabular loss data. In some workflows, triangles may have already
        been created, and are available in external files. In such cases, the
        triangle formatted data is read into a DataFrame, then coerced
        into the desired triangle representation directly. Default value is
        False.

    origin: str
        The field in ``data`` representing origin year. When ``data_shape="triangle"``,
        ``origin`` is ignored. Default value is "origin".

    dev: str
        The field in ``data`` representing development period. When
        ``data_shape="triangle"``,  ``dev`` is ignored. Default value is
        "dev".

    value: str
        The field in ``data`` representing the metric of interest (losses, counts, etc.).
        When ``data_shape="triangle"``, ``value`` is ignored. Default value is "value".

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
            raise NameError("Invalid data_format argument: `{}`.".format(tri_type))

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
    if tri_type.lower().startswith("i"):
        tri = IncrTriangle(data=df, origin=origin, dev=dev, value=value)

    elif tri_type.lower().startswith("c"):
        tri = CumTriangle(data=df, origin=origin, dev=dev, value=value)

        # Replace missing actuals.
        for origin_ in tri.index:
            origin_indx = tri.index.get_loc(origin_)
            origin_init_val = tri.iat[origin_indx, 0]
            if np.isnan(origin_init_val):
                tri.iat[origin_indx, 0] = 1.

            for devp_indx, devp_ in enumerate(tri.columns[1:], start=1):
                triind_val = tri.triind.iat[origin_indx, devp_indx]
                if triind_val==0:
                    if np.isnan(tri.iat[origin_indx, devp_indx]):
                        tri.iat[origin_indx, devp_indx] = tri.iat[origin_indx, (devp_indx - 1)]
    return(tri)


