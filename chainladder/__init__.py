"""
This module contains the class definitions for ``_BaseChainLadder``,
``_MackChainLadder`` and ``_BootstrapChainLadder``. Users should avoid
calling any ``*ChainLadder`` class constructors directly; rather the dataset
and triangle arguments should be passed to ``chladder``, which will return
the initialized ChainLadder instance, from which estimates of outstanding
liabilities and optionally ranges can be obtained.
"""
import pandas as pd
import numpy as np


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
    def __init__(self, cumtri, sel="all-weighted", tail=1.0):
        """
        Generate point estimates for outstanding claim liabilities at
        ultimate for each origin year and in aggregate. The
        ``BaseChainLadder`` class exposes no functionality to estimate
        variability around the point estimates at ultimate.

        Parameters
        ----------
        cumtri: triangle._CumTriangle
            A cumulative.CumTriangle instance.

        sel: str
            Specifies which set of age-to-age averages should be specified as
            the chain ladder loss development factors (LDFs). All available
            age-to-age averages can be obtained by calling
            ``self.tri.a2a_avgs``. Default value is "all-weighted".

        tail: float
            The Chain Ladder tail factor, which accounts for development
            beyond the last development period. Defaults to 1.0.
        """
        self.tri  = cumtri
        self.sel  = sel
        self.tail = tail

        # Properties
        self._ldfs = None
        self._cldfs = None
        self._ultimates = None
        self._reserves = None




    def __call__(self):
        """
        Return a summary of ultimate and reserve estimates resulting from
        the application of the development technique over self.tri. Summary
        DataFrame is comprised of origin year, maturity of origin year, loss
        amount at latest evaluation, cumulative loss development factors,
        projected ultimates and the reserve estimate, by origin year and in
        aggregate.

        Returns
        -------
        pd.DataFrame
            Summary with values by origin year for maturity, latest cumulative
            loss amount, cumulative loss development factor, projected
            ultimate loss and the reserve amount.
        """
        summcols = ["maturity", "latest", "cldf", "ultimate", "reserve"]
        summDF   = pd.DataFrame(columns=summcols, index=self.tri.index)
        summDF["maturity"]  = self.tri.maturity.astype(np.str)
        summDF["latest"]    = self.tri.latest_by_origin
        summDF["cldf"]      = self.cldfs.values[::-1]
        summDF["ultimate"]  = self.ultimates
        summDF["reserve"]   = self.reserves
        summDF.loc["total"] = summDF.sum()

        # Set to NaN columns that shouldn't be summed.
        summDF.loc["total", "maturity"] = ""
        summDF.loc["total", "cldf"]     = np.NaN
        summDF = summDF.reset_index().rename({"index":"origin"}, axis="columns")
        return(summDF)


    @property
    def ldfs(self):
        """
        Lookup the loss development factors associated with ``self.sel``.

        Returns
        -------
        pd.Series
            Loss development factors as pd.Series.
        """
        if self._ldfs is None:
            try:
                ldfsinit   = self.tri.a2a_avgs.loc[self.sel].astype(np.float_)
                self._ldfs = ldfsinit.append(pd.Series(data=[self.tail], index=["tail"]))
            except KeyError:
                print("Invalid age-to-age average selected: `{}`".format(self.sel))
        return(self._ldfs.astype(np.float_).sort_index())



    @ldfs.setter
    def ldfs(self, update_spec):
        """
        Update/override default Chaion Ladder loss development factors.

        Parameters
        ----------
        update_spec: tuple
            2-tuple consisting of ``(index, value)``, representing
            the index of the target cell in self.ldfs , and the value
            used to update it.
        """
        indx, value = update_spec
        if indx in self._ldfs.index:
            self._ldfs[indx] = value



    @property
    def cldfs(self):
        """
        Calculate cumulative loss development factors factors (cldfs) by
        successive multiplication beginning with the tail factor and the
        oldest age-to-age factor. The cumulative claim development factor
        projects the total growth over the remaining valuations. Cumulative
        claim development factors are also known as "Age-to-Ultimate Factors"
        or "Claim Development Factors to Ultimate".

        Returns
        -------
        pd.Series
            Cumulative loss development factors as pd.Series.
        """
        cldfs_index = self.ldfs.index.values
        cldfs_vals  = np.cumprod(self.ldfs.values[::-1])[::-1]
        self._cldfs = pd.Series(data=cldfs_vals, index=cldfs_index, name="cldfs")
        return(self._cldfs.astype(np.float_).sort_index())



    @cldfs.setter
    def cldfs(self, update_spec):
        """
        Update/override default Chaion Ladder cumulative loss development
        factors.

        Parameters
        ----------
        update_spec: tuple
            2-tuple consisting of ``(index, value)``, representing
            the index of the target value in self.cldfs , and
            the value used to update it.
        """
        indx, value = update_spec
        if indx in self._cldfs.index:
            self._cldfs[indx] = value



    @property
    def ultimates(self) -> np.ndarray:
        """
        Ultimate claims are equal to the product of the latest valuation of
        losses (the amount along latest diagonal of any ``CumTriangle``
        instance) and the appropriate cldf/age-to-ultimate factor. We
        determine the appropriate age-to-ultimate factor based on the age
        of each origin year relative to the evaluation date.

        Returns
        -------
        pd.Series
            Ultimate loss projections by origin year as pd.Series.
        """
        self._ultimates = pd.Series(
            data=self.tri.latest_by_origin.values * self.cldfs.values[::-1],
            index=self.tri.index, name="ultimates")
        return(self._ultimates.astype(np.float_).sort_index())


    @property
    def reserves(self) -> np.ndarray:
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

        Returns
        -------
        pd.Series
            The difference between the ultimate loss projections and
            cumulative loss amounts at the latest evaluation period for each
            origin year.
        """
        self._reserves = pd.Series(
            data=self.ultimates - self.tri.latest_by_origin,
            index=self.tri.index, name='reserves')
        return(self._reserves.astype(np.float_).sort_index())


    @property
    def trisqrd(self) -> np.ndarray:
        """
        Project claims growth for each future development period. Returns a
        DataFrame of loss projections for each subsequent development period
        for each accident year. Populates the triangle's lower-right or
        southeast portion (i.e., "squaring the triangle").

        Returns
        -------
        pd.DataFrame
            Table of "squared" triangle as pd.DataFrame.
        """
        self._trisqrd = self.tri.copy(deep=True)
        ldfs  = self.ldfs.values
        rposf = self.tri.index.size
        clvi  = self.tri.clvi["row_offset"]
        for i in enumerate(self._trisqrd.columns[1:], start=1):
            ii  , devp  = i[0], i[1]
            ildf, rposi = ldfs[ii - 1], clvi[devp] + 1
            self._trisqrd.iloc[rposi:rposf, ii] = \
                self._trisqrd.iloc[rposi:rposf, ii - 1] * ildf
        # Multiply right-most column by tail factor.
        max_devp = self._trisqrd.columns[-1]
        self._trisqrd["ultimate"] = self._trisqrd.loc[:,max_devp].values * self.tail
        return(self._trisqrd.astype(np.float_).sort_index())
