"""
_MackChainLadder implementation.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from . import BaseChainLadder



class MackChainLadder(BaseChainLadder):
    """
    Perform Mack Chain Ladder method. The predicition variance is comprised
    of the estimation variance and the process variance. Estimation variance
    arises from the inability to accurately define the distribution from which
    past events have been generated. Process variance arises from the
    inability to accurately predict which single outcome from the distribution
    will occur at a given time. The predicition error is defined as the
    standard deviation of the forecast.

    References
    ----------
    - Mack, Thomas (1993) *Measuring the Variability of Chain Ladder Reserve
      Estimates*, 1993 CAS Prize Paper Competition on 'Variability of Loss Reserves'.

    - Mack, Thomas, (1993), *Distribution-Free Calculation of the Standard Error
      of Chain Ladder Reserve Estimates*, ASTIN Bulletin 23, no. 2:213-225.

    - Mack, Thomas, (1999), *The Standard Error of Chain Ladder Reserve Estimates:
      Recursive Calculation and Inclusion of a Tail Factor*, ASTIN Bulletin 29,
      no. 2:361-366.

    - England, P., and R. Verrall, (2002), *Stochastic Claims Reserving in General
      Insurance*, British Actuarial Journal 8(3): 443-518.

    - Murphy, Daniel, (2007), *Chain Ladder Reserve Risk Estimators*, CAS E-Forum,
      Summer 2007.
    """
    def __init__(self, cumtri):
        """
        Parameters
        ----------
        cumtri: triangle._CumTriangle
            A cumulative.CumTriangle instance
        """

        super().__init__(cumtri)

        # Properties.
        self._mod_a2aind = None
        self._mod_tri = None



    def __call__(self, alpha=1, tail=1.0, dist="lnorm", q=[.75, .95], two_sided=False):
        """
        Return a summary of ultimate and reserve estimates resulting from
        the application of the development technique over self.tri. Summary
        DataFrame is comprised of origin year, maturity of origin year, loss
        amount at latest evaluation, cumulative loss development factors,
        projected ultimates and the reserve estimate, by origin year and in
        aggregate.

        ### TODO ###
        Implement tail factor functionality.

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
        """
        ldfs = self._ldfs(alpha=alpha)
        cldfs = self._cldfs(ldfs=ldfs, tail=1.0)
        maturity = self.tri.maturity.astype(np.str)
        latest = self.tri.latest_by_origin
        ults = self._ultimates(cldfs=cldfs)
        ibnr = self._reserves(ultimates=ults)
        devpvar = self._devp_variance(ldfs=ldfs, alpha=alpha)
        ldfvar = self._ldf_variance(devpvar=devpvar, alpha=alpha)
        proc_error = pd.Series(
            self._process_error(ldfs=ldfs, devpvar=devpvar).iloc[:,-1].replace(np.NaN, 0),
            name="process_error", dtype=np.float
            )
        param_error = pd.Series(
            self._parameter_error(ldfs=ldfs, ldfvar=ldfvar).iloc[:,-1].replace(np.NaN, 0),
            name="parameter_error", dtype=np.float
            )
        mse = self._mean_squared_error(process_error=proc_error, parameter_error=param_error)
        std_error = pd.Series(np.sqrt(mse), name="std_error")
        cv = pd.Series(std_error / ibnr, name="cv")

        dfmatur = maturity.to_frame().reset_index(drop=False).rename({"index":"origin"}, axis=1)
        dfcldfs = cldfs.to_frame().reset_index(drop=False).rename({"index":"maturity"}, axis=1)
        dfcldfs["maturity"] = dfcldfs["maturity"].astype(np.str)
        dfsumm = dfmatur.merge(dfcldfs, on=["maturity"], how="left").set_index("origin")
        dfsumm.index.name = None
        dflatest = latest.to_frame().rename({"latest_by_origin":"latest"}, axis=1)
        dfsumm = functools.reduce(
            lambda df1, df2: df1.join(df2),
            (dflatest, ults.to_frame(), ibnr.to_frame(), std_error.to_frame(), cv.to_frame()),
            dfsumm
            )

        if dist=="norm":
            std_params, mean_params = std_error, ibnr

            def rv(mu, sigma, z):
                """
                Return evaluated Normal distribution with mean ``mu`` and
                standard deviation ``sigma`` at ``z``.
                """
                return(mu + sigma * z)

        if dist=="lognorm":
            std_params = np.sqrt(np.log(1 + (std_error/ ibnr)**2))
            mean_params = np.log(ibnr) - .50 * std_params**2

            def rv(mu, sigma, z):
                """
                Return evaluated Log-normal distribution with mean ``mu`` and
                standard deviation ``sigma`` at ``z``.
                """
                return(np.exp(mu + sigma * z))



        q = [.75, .95]
        two_sided = False


        qtls = np.asarray([q] if isinstance(q, (float, int)) else q)
        if np.all(np.logical_and(qtls <= 1, qtls >= 0)):
            if two_sided:
                qtls = np.sort(np.unique(np.append((1 - qtls) / 2., (1 + qtls) / 2.)))
            else:
                qtls = np.sort(np.unique(qtls))
        else:
            raise ValueError("Values for quantiles must fall between [0, 1].")

        qtlsfmt = ["{:.5f}".format(i).rstrip("0").rstrip(".") + "%" for i in 100 * qtls]

        for ii, jj in zip(qtls, qtlsfmt):


            dfsumm[pctlstr_] = dfsumm.index.map(
                lambda v: np.percentile(
                    dfreserves[dfreserves["origin"]==v]["reserve"].values,
                    100 * q_, interpolation=interpolation
                )
            )
        for ii in q:

            qtls = np.asarray([q] if isinstance(q, (float, int)) else q)

            ii = .95

            Z = norm().ppf(ii)

            print(Z)









        # Compute correlation term for aggregate reserve estimate.


        #
        # sigma_i = np.log(1 + self.msepi / ibnr**2)
        # mu_i    = np.log(ibnr - .50 * sigma_i)
        #
        # summcols = ["maturity", "latest", "cldf", "ultimate", "reserve"]
        # summDF   = pd.DataFrame(columns=summcols, index=self.tri.index)
        # summDF["maturity"]  = self.tri.maturity.astype(np.str)
        # summDF["latest"]    = self.tri.latest_by_origin
        # summDF["cldf"]      = cldfs.values[::-1]
        # summDF["ultimate"]  = ults
        # summDF["reserve"]   = ibnr
        # self._summary['RMSEP']        = self.rmsepi
        # self._summary['CV']           = self.rmsepi/self.reserves
        # self._summary["NORM_95%_LB"]  = self.reserves - (1.96 * self.rmsepi)
        # self._summary["NORM_95%_UB"]  = self.reserves + (1.96 * self.rmsepi)
        # self._summary["LNORM_95%_LB"] = np.exp(mu_i - 1.96 * sigma_i)
        # self._summary["LNORM_95%_UB"] = np.exp(mu_i + 1.96 * sigma_i)
        # self._summary.loc['TOTAL']    = self._summary.sum()
        # summDF.loc["total"] = summDF.sum()
        #
        # # Set to NaN columns that shouldn't be summed.
        # summDF.loc["total", "maturity"] = ""
        # summDF.loc["total", "cldf"]     = np.NaN
        # summDF = summDF.reset_index().rename({"index":"origin"}, axis="columns")
        #
        #
        #
        #
        #     # # Populate self._summary with existing properties if available.
        #     # self._summary['LATEST']       = self.latest_by_origin
        #     # self._summary['CLDF']         = self.cldfs[::-1]
        #     # self._summary['EMERGENCE']    = 1/self.cldfs[::-1]
        #     # self._summary['ULTIMATE']     = self.ultimates
        #     # self._summary['RESERVE']      = self.reserves
        #     # self._summary['RMSEP']        = self.rmsepi
        #     # self._summary['CV']           = self.rmsepi/self.reserves
        #     # self._summary["NORM_95%_LB"]  = self.reserves - (1.96 * self.rmsepi)
        #     # self._summary["NORM_95%_UB"]  = self.reserves + (1.96 * self.rmsepi)
        #     # self._summary["LNORM_95%_LB"] = np.exp(mu_i - 1.96 * sigma_i)
        #     # self._summary["LNORM_95%_UB"] = np.exp(mu_i + 1.96 * sigma_i)
        #     # self._summary.loc['TOTAL']    = self._summary.sum()
        #     #
        #     # # Set CLDF Total value to `NaN`.
        #     # self._summary.loc["TOTAL","CLDF","EMERGENCE"] = np.NaN
        #
        # return(self._summary)


        # pctl_ = np.asarray([q] if isinstance(q, (float, int)) else q)
        #
        # if np.any(np.logical_or(pctl_ <= 1, pctl_ >= 0)):
        #     if two_sided:
        #         pctlarr = np.sort(np.unique(np.append((1 - pctl_) / 2., (1 + pctl_) / 2.)))
        #     else:
        #         pctlarr = np.sort(np.unique(pctl_))
        # else:
        #     raise ValueError("Values for quantiles must fall between [0, 1].")
        #
        # # Compile Chain Ladder point estimate summary.
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
        # )
        #
        # dfsumm = dfsumm.rename({"reserve":"cl_reserve"}, axis=1)
        #
        # # Compute median distribution of bootstrap samples.
        # dfsumm["bcl_reserve"] = dfsumm.index.map(
        #     lambda v: np.percentile(
        #         dfreserves[dfreserves["origin"]==v]["reserve"].values,
        #         100 * .50, interpolation=interpolation
        #     )
        # )
        #
        # # Attach percentile fields to dfsumm.
        # pctlfmt = ["{:.5f}".format(i).rstrip("0").rstrip(".") + "%" for i in 100 * pctlarr]
        # for q_, pctlstr_ in zip(pctlarr, pctlfmt):
        #     dfsumm[pctlstr_] = dfsumm.index.map(
        #         lambda v: np.percentile(
        #             dfreserves[dfreserves["origin"]==v]["reserve"].values,
        #             100 * q_, interpolation=interpolation
        #         )
        #     )
        #
        # # Add "Total" index and set to NaN fields that shouldn't be aggregated.
        # dfsumm.loc["total"] = dfsumm.sum()
        # dfsumm.loc["total", "maturity"] = ""
        # dfsumm.loc["total", "cldf"] = np.NaN
        #
        # dfsumm = dfsumm.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        #
        # kwds = {"sel":"all-weighted", "sims": sims, "neg_handler":neg_handler,
        #         "procdist":procdist, "parametric":parametric,
        #         "q":q, "interpolation":interpolation,}
        #
        # sampling_dist_res = None if parametric==True else sampling_dist_
        #
        # bclresult = BootstrapChainLadderResult(
        #     summary=dfsumm, reserve_dist=dfreserves, sims_data=dfprocerror,
        #     tri=self.tri, ldfs=ldfs_, cldfs=cldfs_, latest=latest_,
        #     maturity=maturity_, ultimates=ultimates_, reserves=reserves_,
        #     scale_param=scale_param, unscaled_residuals=unscld_residuals_,
        #     adjusted_residuals=adjust_residuals_, sampling_dist=sampling_dist_res,
        #     fitted_tri_cum=tri_fit_cum_, fitted_tri_incr=tri_fit_incr_,
        #     trisqrd=trisqrd_, **kwds)
        #


    @staticmethod
    def get_quantile(*pctl):
        pass


    @property
    def mod_tri(self):
        """
        Return modified triangle-shaped DataFrame with same indices as
        self.tri.a2a.

        Returns
        -------
        pd.DataFrame
        """
        if self._mod_tri is None:
            self._mod_tri = self.tri.copy(deep=True)
            for ii in range(self.tri.latest.shape[0]):
                r_indx = self.tri.latest.loc[ii, "origin"].item()
                c_indx = self.tri.latest.loc[ii, "dev"].item()
                self._mod_tri.at[r_indx, c_indx] = np.NaN
            self._mod_tri = self._mod_tri.dropna(axis=0, how="all").dropna(axis=1, how="all")
        return(self._mod_tri)


    @property
    def mod_a2aind(self):
        """
        Return self.tri.a2aind with lower-right 0s replaced with NaN.

        Returns
        -------
        pd.DataFrame
        """
        if self._mod_a2aind is None:
            self._mod_a2aind = self.tri.a2aind.replace(0, np.NaN)
        return(self._mod_a2aind)


    def _ldfs(self, alpha=1):
        """
        Compute Mack loss development factors.

        Parameters
        ----------
        alpha: {0, 1, 2}
            * ``0``: Straight average of observed individual link ratios.
            * ``1``: Historical Chain Ladder age-to-age factors.
            * ``2``: Regression of $C_{k+1}$ on $C_{k}$ with 0 intercept.

        tail: float
            Tail factor. At present, must be 1.0. This will change in a
            future release.

        Returns
        -------
        pd.Series
        """
        C, w = self.mod_tri, self.mod_a2aind
        # ldfs = (self.tri.a2a * w * C**alpha).sum(axis=0) / (w * C**alpha).sum(axis=0)
        return((self.tri.a2a * w * C**alpha).sum(axis=0) / (w * C**alpha).sum(axis=0))


    def _cldfs(self, ldfs, tail=1.0):
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

        tail: float
            Tail factor. At present, must be 1.0. This will change in a
            future release.

        Returns
        -------
        pd.Series
        """
        # Determine increment for tail factor development period.
        # # ldfs = ldfs.copy(deep=True)
        increment = np.unique(ldfs.index[1:] - ldfs.index[:-1])[0]
        # ldfs[ldfs.index.max() + increment] = tail
        cldfs_indx = np.append(ldfs.index.values, ldfs.index.max() + increment)
        cldfs = np.cumprod(np.append(ldfs.values, tail)[::-1])[::-1]
        cldfs = pd.Series(data=cldfs, index=cldfs_indx, name="cldf")
        return(cldfs.astype(np.float).sort_index())


    def _ldf_variance(self, devpvar, alpha=1):
        """
        Compute the variance of a given development period's link ratios
        vs. provided ldfs.

        devpvar: pd.Series
            The development period variance, usually represented as
            $\hat{\sigma}^{2}_{k}$ in the literature. For a triangle with
            ``n`` development periods, devpvar will contain ``n-1`` elements.

        alpha: {0, 1, 2}
            * ``0``: Straight average of observed individual link ratios.
            * ``1``: Historical Chain Ladder age-to-age factors.
            * ``2``: Regression of $C_{k+1}$ on $C_{k}$ with 0 intercept.

        Returns
        -------
        pd.Series
        """
        ldfvar = pd.Series(index=devpvar.index, dtype=np.float, name="ldfvar")
        C, w = self.mod_tri, self.mod_a2aind
        for devp in w.columns:
            ldfvar[devp] = devpvar[devp] / (w.loc[:,devp] * C.loc[:, devp]**alpha).sum()
        return(ldfvar)


    def _devp_variance(self, ldfs, alpha=1):
        """
        Compute the development period variance, usually represented as
        $\hat{\sigma}^{2}_{k}$ in the literature. For a triangle with
        ``n`` development periods, result will contain ``n-1`` elements.

        Parameters
        ----------
        ldfs: pd.Series
            Selected ldfs, typically the output of calling ``self._ldfs``,
            or a series of values indexed by development period.

         alpha: {0, 1, 2}
            * ``0``: Straight average of observed individual link ratios.
            * ``1``: Historical Chain Ladder age-to-age factors.
            * ``2``: Regression of $C_{k+1}$ on $C_{k}$ with 0 intercept.

        Returns
        -------
        pd.Series
        """
        devpvar = pd.Series(index=ldfs.index, dtype=np.float, name="devpvar")
        C, w, F = self.mod_tri, self.mod_a2aind, self.tri.a2a
        n = self.tri.origins.size
        for indx, jj in enumerate(self.tri.devp[:-2]):
            devpvar[jj] = \
                (w[jj] * (C[jj]**alpha) * (F[jj] - ldfs[jj])**2).sum() / (n - indx - 2)

        # Calculate development period variance for period n-1.
        devpvar.iloc[-1] = np.min((
            devpvar.iloc[-2]**2 / devpvar.iloc[-3],
            np.min([devpvar.iloc[-2], devpvar.iloc[-3]])
            ))
        return(devpvar)


    def _process_error(self, ldfs, devpvar):
        """
        Return a triangle-shaped DataFrame containing elementwise process
        error. To obtain the process error for a given origin period,
        cells are aggregated across columns.

        Parameters
        ----------
        ldfs: pd.Series
            Selected ldfs, typically the output of calling ``self._ldfs``,
            or a series of values indexed by development period.

        devpvar: pd.Series
            The development period variance, usually represented as
            $\hat{\sigma}^{2}_{k}$ in the literature. For a triangle with
            ``n`` development periods, devpvar will contain ``n-1`` elements.

        Returns
        -------
        pd.DataFrame
        """
        # Create latest reference using 1-based indexing.
        latest = self.tri.latest.sort_index()
        latest.origin = range(1, latest.index.size + 1)
        latest.dev = range(latest.index.size, 0, -1)

        # Create DataFrame to hold cellwise process error.
        dfpe = pd.DataFrame(columns=self.tri.columns, index=self.tri.index)
        dfpe.index = range(1, dfpe.index.size + 1)
        dfpe.columns = range(1, dfpe.columns.size + 1)

        # Bind reference to squared triangle.
        trisqrd = self._trisqrd(ldfs).drop("ultimate", axis=1)
        trisqrd.index = range(1, trisqrd.index.size + 1)
        trisqrd.columns = range(1, trisqrd.columns.size + 1)
        n = self.tri.devp.size

        # `ii` iterates by origin, `kk` by development period.
        for ii in dfpe.index[1:]:
            latest_devp = latest[latest["origin"]==ii]["dev"].item()
            latest_cum = trisqrd.at[ii, latest_devp]
            kk0 = n + 2 - ii
            dfpe.at[ii, kk0] = latest_cum * devpvar.iloc[kk0 - 2]

            for kk in range(kk0 + 1, n + 1):
                term0 = (ldfs.iloc[kk - 2]**2) * dfpe.at[ii, kk - 1]
                term1 = trisqrd.at[ii, kk - 1] * devpvar.iloc[kk - 2]
                dfpe.at[ii, kk] = term0 + term1

        # Re-index dfpe to match self.tri.
        dfpe.columns, dfpe.index = self.tri.columns, self.tri.index
        return(dfpe)


    def _parameter_error(self, ldfs, ldfvar):
        """
         Return a triangle-shaped DataFrame containing elementwise parameter
        error. To obtain the parameter error for a given origin period,
        cells are aggregated across columns.

        Parameters
        ----------
        ldfs: pd.Series
            Selected ldfs, typically the output of calling ``self._ldfs``,
            or a series of values indexed by development period.

        ldfvar: pd.Series
            Link ratio variance. For a triangle with ``n`` development
            periods, ldfvar will contain ``n-1`` elements.

        Returns
        -------
        pd.DataFrame
        """
        # Create latest reference using 1-based indexing.
        latest = self.tri.latest.sort_index()
        latest.origin = range(1, latest.index.size + 1)
        latest.dev = range(latest.index.size, 0, -1)

        # Create DataFrame to hold cellwise parameter error.
        dfpe = pd.DataFrame(columns=self.tri.columns, index=self.tri.index)
        dfpe.index = range(1, dfpe.index.size + 1)
        dfpe.columns = range(1, dfpe.columns.size + 1)

        # Bind reference to squared triangle.
        trisqrd = self._trisqrd(ldfs).drop("ultimate", axis=1)
        trisqrd.index = range(1, trisqrd.index.size + 1)
        trisqrd.columns = range(1, trisqrd.columns.size + 1)
        n = self.tri.devp.size

        # `ii` iterates by origin, `kk` by development period.
        for ii in dfpe.index[1:]:
            latest_devp = latest[latest["origin"]==ii]["dev"].item()
            latest_cum = trisqrd.at[ii, latest_devp]
            kk0 = n + 2 - ii
            dfpe.at[ii, kk0] = latest_cum**2 * ldfvar.iloc[kk0 - 2]

            for kk in range(kk0 + 1, n + 1):

                term0 = (ldfs.iloc[kk - 2]**2) * dfpe.at[ii, kk - 1]
                term1 = (trisqrd.at[ii, kk - 1]**2) * ldfvar.iloc[kk - 2]
                term2 = ldfvar.iloc[kk - 2] * dfpe.at[ii, kk - 1]
                dfpe.at[ii, kk] = term0 + term1 + term2

        # Re-index dfpe to match self.tri.
        dfpe.columns, dfpe.index = self.tri.columns, self.tri.index
        return(dfpe)


    def _mean_squared_error(self, process_error, parameter_error):
        """
        Compute the mean squared error of reserve estimates for each
        origin period.

        Parameters
        ----------
        process_error: pd.Series
            Reserve estimate process error indexed by origin.

        parameter_error: pd.Series
            Reserve estimate parameter error indexed by origin.

        Returns
        -------
        pd.Series
        """
        return(pd.Series(process_error + parameter_error, name="mse"))





    # @property
    # def originref(self):
    #     """
    #     Intended for internal use only. Contains data by origin year.
    #     """
    #     if self._originref is None:
    #         self._originref = pd.DataFrame({
    #             'reserve'      :self.reserves,
    #             'ultimate'     :self.ultimates,
    #             'process_error':self.process_error,
    #             'param_error'  :self.parameter_error,
    #             'msep'         :self.msepi,
    #             'rmsep'        :self.rmsepi}, index=self.tri.index)
    #         self._originref = \
    #             self._originref[
    #                 ["reserve","ultimate","process_error","param_error","msep","rmsep"]
    #                 ]
    #     return(self._originref)
    #
    #
    #
    # @property
    # def devpref(self):
    #     """
    #     Intended for internal use only. Contains data by development period.
    #     """
    #     if self._devpref is None:
    #         self._devpref = pd.DataFrame({
    #             "ldf"    :self.ldfs,
    #             "sse"    :self.devpvar.values,
    #             "ratio"  :(self.devpvar.values / self.ldfs ** 2),
    #             "dev"    :self.tri.columns[:-1],
    #             "inv_sum":self.inverse_sums},
    #             index=self.tri.columns[:-1]
    #             )
    #
    #         self._devpref["indx"] = \
    #             self._devpref["dev"].map(
    #                 lambda x: self.tri.columns.get_loc(x))
    #
    #         self._devpref = \
    #             self._devpref[["dev","indx","ldf","sse","ratio","inv_sum"]]
    #
    #     return(self._devpref)
    #
    #
    #
    # @property
    # def inverse_sums(self):
    #     """
    #     Convenience aggregation for use in parameter error
    #     calcuation.
    #     """
    #     if self._inverse_sums is None:
    #         devp_sums = \
    #             self.tri.sum(axis=0)-self.tri.latest_by_origin[::-1].values
    #         self._inverse_sums = pd.Series(
    #             data=devp_sums,index=devp_sums.index,name='inverse_sums')
    #         self._inverse_sums = (1 / (self._inverse_sums))[:-1]
    #     return(self._inverse_sums)



    # @property
    # def inverse_sums2(self):
    #     """
    #     Convenience aggregation for use in parameter error
    #     calcuation.
    #     """
    #     if self._inverse_sums is None:
    #         devp_sums = list()
    #         for devp in self.tri.columns[:-1]:
    #             iterep  = self.tri.index.get_loc(self.tri[devp].last_valid_index())
    #             devpos  = self.tri.columns.get_loc(devp)
    #             itersum = self.tri.iloc[:iterep,devpos].sum()
    #             devp_sums.append((devp,(1/itersum)))
    #         indx, vals = zip(*devp_sums)
    #         self._inverse_sums = \
    #             pd.Series(data=vals, index=indx, name='inverse_sums')
    #     return(self._inverse_sums)


    # @property
    # def devpvar(self) -> np.ndarray:
    #     """
    #     devpvar = `development period variance`. Return the variance
    #     of each n-1 development periods as a Series object.
    #     """
    #     if self._devpvar is None:
    #         n = self.tri.columns.size
    #         self._devpvar = np.zeros(n - 1, dtype=np.float_)
    #         for k in range(n - 2):
    #             iter_ses = 0  # `square of standard error`
    #             for i in range(n - (k + 1)):
    #                 c_1, c_2 = self.tri.iloc[i, k], self.tri.iloc[i,k+1]
    #                 iter_ses+=c_1*((c_2 / c_1) - self.ldfs[k])**2
    #             iter_ses = iter_ses/(n-k-2)
    #             self._devpvar[k] = iter_ses
    #
    #             # Calculate standard error for dev period n-1.
    #         self._devpvar[-1] = \
    #             np.min((
    #                 self._devpvar[-2]**2 / self._devpvar[-3],
    #                 np.min([self._devpvar[-2],self._devpvar[-3]])
    #                 ))
    #         self._devpvar = pd.Series(
    #             data=self._devpvar, index=self.tri.columns[:self._devpvar.size],
    #             name="sqrd_std_error")
    #     return(self._devpvar)
    #
    #
    #
    #
    # @property
    # def process_error(self):
    #     """
    #     Process error (forecast error) calculation. The process error
    #     component originates from the stochastic movement of the process.
    #     Returns a pandas Series containing estimates of process variance by
    #     origin year.
    #     """
    #     if self._process_error is None:
    #         lastcol, pelist = self.tri.columns.size-1, list()
    #         for rindx in self.tri.rlvi.index:
    #             iult = self.ultimates[rindx]
    #             ilvi = self.tri.rlvi.loc[rindx,:].col_offset
    #             ilvc = self.tri.rlvi.loc[rindx,:].dev
    #             ipe  = 0
    #             if ilvi<lastcol:
    #                 for dev in self.trisqrd.loc[rindx][ilvi:lastcol].index:
    #                     ipe+=(self.devpref.loc[dev,'RATIO'] / self.trisqrd.loc[rindx,dev])
    #                 ipe*=(iult**2)
    #             pelist.append((rindx,ipe))
    #
    #         # Convert list of tuples into Series object.
    #         indx, vals = zip(*pelist)
    #         self._process_error = \
    #             pd.Series(data=vals, index=indx, name="process_error")
    #     return(self._process_error)
    #
    #
    #
    #
    # @property
    # def parameter_error(self):
    #     """
    #     Estimation error (parameter error) reflects the uncertainty in
    #     the estimation of the parameters.
    #     """
    #     if self._parameter_error is None:
    #         lastcol, pelist = self.tri.columns.size-1, list()
    #         for i in enumerate(self.tri.index):
    #             ii, rindx = i[0], i[1]
    #             iult = self.ultimates[rindx]
    #             ilvi = self.tri.rlvi.loc[rindx,:].col_offset
    #             ilvc = self.tri.rlvi.loc[rindx,:].dev
    #             ipe  = 0
    #             if ilvi<lastcol:
    #                 for k in range(ilvi,lastcol):
    #                     ratio  = self.devpref[self.devpref['indx']==k]['ratio'].values[0]
    #                     invsum = self.devpref[self.devpref['indx']==k]['inv_sum'].values[0]
    #                     ipe+=(ratio * invsum)
    #                 ipe*=iult**2
    #             else:
    #                 ipe = 0
    #             pelist.append((rindx, ipe))
    #         # Convert list of tuples into Series object.
    #         indx, vals = zip(*pelist)
    #         self._parameter_error = \
    #             pd.Series(data=vals, index=indx, name="parameter_error")
    #     return(self._parameter_error)
    #
    #
    #
    #
    # @property
    # def covariance_term(self):
    #     """
    #     Used to derive the conditional mean squared error of total
    #     reserve prediction. MSE_(i,j) is non-zero only for cells
    #     in which i < j (i.e., the
    #     :return:
    #     """
    #     pass
    #
    #
    # @property
    # def msepi(self):
    #     """
    #     Return the mean squared error of predicition by origin year.
    #     Does not contain estimate for total MSEP.
    #     MSE_i = process error + parameter error
    #     """
    #     if self._msepi is None:
    #         self._msepi = self.process_error + self.parameter_error
    #     return(self._msepi)
    #
    #
    # @property
    # def rmsepi(self):
    #     """
    #     Return the root mean squared error of predicition by origin
    #     year. Does not contain estimate for total MSEP.
    #
    #         MSE_i = process error + parameter error
    #     """
    #     if self._rmsepi is None:
    #         self._rmsepi = np.sqrt(self.msepi)
    #     return(self._rmsepi)
    #
    #
    #
    #
    # @property
    # def summary(self):
    #     """
    #     Return a DataFrame containing summary statistics resulting
    #     from applying the development method to tri, in addition
    #     to Mack-generated range estimates.
    #     """
    #     if self._summary is None:
    #         self._summary = \
    #             pd.DataFrame(
    #                 columns=[
    #                     "LATEST","CLDF","EMERGENCE","ULTIMATE","RESERVE","RMSEP",
    #                     "CV","NORM_95%_LB","NORM_95%_UB","LNORM_95%_LB","LNORM_95%_UB"
    #                     ], index=self.tri.index
    #                 )
    #
    #         # Initialize lognormal confidence interval parameters.
    #         sigma_i = np.log(1 + self.msepi/self.reserves**2)
    #         mu_i    = np.log(self.reserves - .50 * sigma_i)
    #
    #         # Populate self._summary with existing properties if available.
    #         self._summary['LATEST']       = self.latest_by_origin
    #         self._summary['CLDF']         = self.cldfs[::-1]
    #         self._summary['EMERGENCE']    = 1/self.cldfs[::-1]
    #         self._summary['ULTIMATE']     = self.ultimates
    #         self._summary['RESERVE']      = self.reserves
    #         self._summary['RMSEP']        = self.rmsepi
    #         self._summary['CV']           = self.rmsepi/self.reserves
    #         self._summary["NORM_95%_LB"]  = self.reserves - (1.96 * self.rmsepi)
    #         self._summary["NORM_95%_UB"]  = self.reserves + (1.96 * self.rmsepi)
    #         self._summary["LNORM_95%_LB"] = np.exp(mu_i - 1.96 * sigma_i)
    #         self._summary["LNORM_95%_UB"] = np.exp(mu_i + 1.96 * sigma_i)
    #         self._summary.loc['TOTAL']    = self._summary.sum()
    #
    #         # Set CLDF Total value to `NaN`.
    #         self._summary.loc["TOTAL","CLDF","EMERGENCE"] = np.NaN
    #
    #     return(self._summary)
    #
    #
    # @staticmethod
    # def get_quantile(*pctl):
    #     pass
    #
    #
    #
    # def __repr__(self):
    #     """
    #     Override default numerical precision used for representing
    #     ultimate loss projections and age-to-ultimate factors.
    #     """
    #     summary_cols = [
    #         "LATEST","CLDF","EMERGENCE","ULTIMATE","RESERVE","RMSEP",
    #         "CV","NORM_95%_LB","NORM_95%_UB","LNORM_95%_LB","LNORM_95%_UB"
    #         ]
    #
    #     summ_specs = pd.Series([0, 5, 5, 0, 0, 0, 5, 0, 0, 0, 0], index=summary_cols)
    #     return(self.summary.round(summ_specs).to_string())
    #
    #
