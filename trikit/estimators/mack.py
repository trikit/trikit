"""
_MackChainLadder implementation.
"""
import functools
import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import norm, lognorm
from scipy.optimize import root
from .base import BaseRangeEstimator, BaseRangeEstimatorResult



class MackChainLadder(BaseRangeEstimator):
    """
    Mack Chain Ladder estimator. The predicition variance is comprised
    of the estimation variance and the process variance. Estimation variance
    arises from the inability to accurately define the distribution from which
    past events have been generated. Process variance arises from the
    inability to accurately predict which single outcome from the distribution
    will occur at a given time. The predicition error is defined as the
    standard deviation of the forecast.

    Parameters
    ----------
    cumtri: triangle._CumTriangle
        A cumulative.CumTriangle instance

    References
    ----------
    1. Mack, Thomas (1993) *Measuring the Variability of Chain Ladder Reserve
       Estimates*, 1993 CAS Prize Paper Competition on Variability of Loss Reserves.

    2. Mack, Thomas, (1993), *Distribution-Free Calculation of the Standard Error
       of Chain Ladder Reserve Estimates*, ASTIN Bulletin 23, no. 2:213-225.

    3. Mack, Thomas, (1999), *The Standard Error of Chain Ladder Reserve Estimates:
       Recursive Calculation and Inclusion of a Tail Factor*, ASTIN Bulletin 29,
       no. 2:361-366.

    4. England, P., and R. Verrall, (2002), *Stochastic Claims Reserving in General Insurance*, British Actuarial Journal 8(3): 443-518.

    5. Murphy, Daniel, (2007), *Chain Ladder Reserve Risk Estimators*, CAS E-Forum, Summer 2007.

    6. Carrato, A., McGuire, G. and Scarth, R. 2016. *A Practitioner's
       Introduction to Stochastic Reserving*, The Institute and Faculty of
       Actuaries. 2016.
    """
    def __init__(self, cumtri):

        super().__init__(cumtri)

        # Properties.
        self._mod_a2aind = None
        self._mod_tri = None



    def __call__(self, alpha=1, tail=1.0, dist="lognorm", q=[.75, .95], two_sided=False):
        """
        Return a summary of ultimate and reserve estimates resulting from
        the application of the Mack Chain Ladder over self.tri. Summary
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
        """
        ldfs = self._ldfs(alpha=alpha, tail=1.0)
        cldfs = self._cldfs(ldfs=ldfs)
        maturity = self.tri.maturity.astype(np.str)
        latest = self.tri.latest_by_origin
        ultimates = self._ultimates(cldfs=cldfs)
        reserves = self._reserves(ultimates=ultimates)
        devpvar = self._devp_variance(ldfs=ldfs, alpha=alpha)
        ldfvar = self._ldf_variance(devpvar=devpvar, alpha=alpha)
        proc_error = pd.Series(
            self._process_error(ldfs=ldfs, devpvar=devpvar).iloc[:, -1].replace(np.NaN, 0),
            name="process_error", dtype=np.float
            )
        param_error = pd.Series(
            self._parameter_error(ldfs=ldfs, ldfvar=ldfvar).iloc[:, -1].replace(np.NaN, 0),
            name="parameter_error", dtype=np.float
            )
        mse = self._mean_squared_error(
            process_error=proc_error, parameter_error=param_error
            )
        std_error = pd.Series(np.sqrt(mse), name="std_error")
        cv = pd.Series(std_error / reserves, name="cv")
        trisqrd = self._trisqrd(ldfs=ldfs)

        dfmatur = maturity.to_frame().reset_index(drop=False).rename({"index": "origin"}, axis=1)
        dfcldfs = cldfs.to_frame().reset_index(drop=False).rename({"index": "maturity"}, axis=1)
        dfcldfs["maturity"] = dfcldfs["maturity"].astype(str)
        dfcldfs["emergence"] = 1 / dfcldfs["cldf"]
        dfsumm = dfmatur.merge(dfcldfs, on=["maturity"], how="left").set_index("origin")
        dfsumm.index.name = None
        dflatest = latest.to_frame().rename({"latest_by_origin": "latest"}, axis=1)
        dfsumm = functools.reduce(
            lambda df1, df2: df1.join(df2),
            (dflatest, ultimates.to_frame(), reserves.to_frame(), std_error.to_frame(), cv.to_frame()),
            dfsumm
            )

        # Add total index and set to NaN fields that shouldn't be aggregated.
        dfsumm.loc["total"] = dfsumm.sum()
        dfsumm.loc["total", "maturity"] = ""
        dfsumm.loc["total", ["cldf", "emergence"]] = np.NaN

        # Create latest and trisqrd reference using 1-based indexing.
        latest = self.tri.latest.sort_index()
        latest.origin = range(1, latest.index.size + 1)
        latest.dev = range(latest.index.size, 0, -1)
        trisqrd = self._trisqrd(ldfs).drop("ultimate", axis=1)
        trisqrd.index = range(1, trisqrd.index.size + 1)
        trisqrd.columns = range(1, trisqrd.columns.size + 1)

        # Compute mse for aggregate reserve.
        n = self.tri.devp.size
        mse_total = pd.Series(index=dfsumm.index[:-1], dtype=np.float)
        quotient = pd.Series(devpvar / ldfs**2, dtype=np.float).reset_index(drop=True)
        quotient.index = quotient.index + 1
        for indx, ii in enumerate(mse_total.index[1:], start=2):
            mse_ii, ult_ii = mse[ii], ultimates[ii]
            ults_sum = ultimates[ultimates.index > ii].dropna().sum()
            rh_sum = sum(
                quotient[jj] / sum(trisqrd.loc[mm, jj]
                    for mm in range(1, (n - jj) + 1)) for jj in range(n + 1 - indx, n)
                    )
            mse_total[ii] = mse_ii + 2 * ult_ii * ults_sum * rh_sum

        # Reset trisqrd columns and index back to original values.
        trisqrd.columns, trisqrd.index = self.tri.columns, self.tri.index
        dfsumm.loc["total", "std_error"] = np.sqrt(mse_total.dropna().sum())
        dfsumm.loc["total", "cv"] = dfsumm.loc["total", "std_error"] / dfsumm.loc["total", "reserve"]

        if dist == "norm":
            std_params, mean_params = dfsumm["std_error"], dfsumm["reserve"]
            rv_list = [norm(loc=ii, scale=jj) for ii, jj in zip(mean_params, std_params)]

        elif dist == "lognorm":
            with np.errstate(divide="ignore"):
                std_params = np.sqrt(np.log(1 + (dfsumm["std_error"] / dfsumm["reserve"])**2)).replace(np.NaN, 0)
                mean_params = np.clip(np.log(dfsumm["reserve"]), a_min=0, a_max=None) - .50 * std_params**2
                rv_list = [lognorm(scale=np.exp(ii), s=jj) for ii, jj in zip(mean_params, std_params)]
        else:
            raise ValueError(
                "dist must be one of {{'norm', 'lognorm'}}, not `{}`.".format(dist)
                )

        rvs = pd.Series(rv_list, index=dfsumm.index)
        qtls, qtlhdrs = self._qtls_formatter(q=q, two_sided=two_sided)

        # Populate qtlhdrs columns with estimated quantile estimates.
        with np.errstate(invalid="ignore"):
            for ii, jj in zip(qtls, qtlhdrs):
                for origin in rvs.index:
                    dfsumm.loc[origin, jj] = rvs[origin].ppf(ii)

        dfsumm.loc[self.tri.index.min(), ["cv"] + qtlhdrs] = np.NaN

        mcl_result = MackChainLadderResult(
            summary=dfsumm, tri=self.tri, ldfs=ldfs, trisqrd=trisqrd, dist=dist,
            process_error=proc_error, parameter_error=param_error, devpvar=devpvar,
            ldfvar=ldfvar, rvs=rvs, alpha=alpha, tail=tail
            )

        return(mcl_result)


    @property
    def mod_tri(self):
        """
        Return modified triangle-shaped DataFrame with same indices as ``self.tri.a2a``.

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
        Return self.tri.a2aind with lower right 0s replaced with NaN.

        Returns
        -------
        pd.DataFrame
        """
        if self._mod_a2aind is None:
            self._mod_a2aind = self.tri.a2aind.replace(0, np.NaN)
        return(self._mod_a2aind)


    def _ldfs(self, alpha=1, tail=1.0):
        """
        Compute Mack loss development factors.

        Parameters
        ----------
        alpha: {0, 1, 2}
            * 0: Straight average of observed individual link ratios.
            * 1: Historical Chain Ladder age-to-age factors.
            * 2: Regression of :math:`C_{k+1}` on :math:`C_{k}` with 0 intercept.

        tail: float
            Tail factor. At present, must be 1.0. This may change in a future release.

        Returns
        -------
        pd.Series
        """
        C, w = self.mod_tri, self.mod_a2aind
        ldfs = (self.tri.a2a * w * C**alpha).sum(axis=0) / (w * C**alpha).sum(axis=0)
        increment = np.unique(ldfs.index[1:] - ldfs.index[:-1])[0]
        ldfs.loc[ldfs.index.max() + increment] = tail
        return(ldfs)


    def _ldf_variance(self, devpvar, alpha=1):
        """
        Compute the variance of a given development period's link ratios
        w.r.t. selected ldfs.

        devpvar: pd.Series
            The development period variance, usually represented as
            :math:`\\hat{\\sigma}^{2}_{k}` in the literature. For a triangle with
            n development periods, devpvar will contain n-1 elements.

        alpha: {0, 1, 2}
            * 0: Straight average of observed individual link ratios.
            * 1: Historical Chain Ladder age-to-age factors.
            * 2: Regression of :math`C_{k+1}` on :math: `C_{k}` with 0 intercept.

        Returns
        -------
        pd.Series
        """
        ldfvar = pd.Series(index=devpvar.index, dtype=np.float, name="ldfvar")
        C, w = self.mod_tri, self.mod_a2aind
        for devp in w.columns:
            ldfvar[devp] = devpvar[devp] / (w.loc[:, devp] * C.loc[:, devp]**alpha).sum()
        return(ldfvar)


    def _devp_variance(self, ldfs, alpha=1):
        """
        Compute the development period variance, usually represented as
        :math:`\\hat{\\sigma}^{2}_{k}` in the literature. For a triangle with
        n development periods, result will contain n-1 elements.

        Parameters
        ----------
        ldfs: pd.Series
            Selected ldfs, typically the output of calling ``self._ldfs``, or a series
            of values indexed by development period.

         alpha: {0, 1, 2}
            * 0: Straight average of observed individual link ratios.
            * 1: Historical Chain Ladder age-to-age factors.
            * 2: Regression of :math:`C_{k+1}` on :math:`C_{k}` with 0 intercept.

        Returns
        -------
        pd.Series
        """
        devpvar = pd.Series(index=ldfs.index[:-1], dtype=np.float, name="devpvar")
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
            :math:`\\hat{\\sigma}^{2}_{k}` in the literature. For a triangle with
            n development periods, devpvar will contain n-1 elements.

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
            latest_devp = latest[latest["origin"] == ii]["dev"].item()
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
            Link ratio variance. For a triangle with n development
            periods, ldfvar will contain n-1 elements.

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
            latest_devp = latest[latest["origin"] == ii]["dev"].item()
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
        origin period. The standard error for each origin period
        is the square root of the mean squared error.


        Parameters
        ----------
        process_error: pd.Series
            Reserve estimate process error indexed by origin. Represents the
            risk associated with the projection of future contingencies that
            are inherently variable, even if the parameters are known
            with certainty.

        parameter_error: pd.Series
            Reserve estimate parameter error indexed by origin. Represents
            the risk that the parameters used in the methods or models are not
            representative of future outcomes.

        Returns
        -------
        pd.Series
        """
        return(pd.Series(process_error + parameter_error, name="mse"))



class MackChainLadderResult(BaseRangeEstimatorResult):
    """
    Container class for ``MackChainLadder`` output.

    Parameters
    ----------
    summary: pd.DataFrame
        ``MackChainLadder`` summary.

    tri: trikit.triangle.CumTriangle
        A cumulative triangle instance.

    alpha: int
        MackChainLadder alpha parameter.

    tail: float
        Tail factor.

    ldfs: pd.Series
        Loss development factors.

    trisqrd: pd.DataFrame
        Projected claims growth for each future development period.

    dist: str
        The distribution function chosen to approximate the true distribution of
        reserves by origin period. Wither "norm" or "lognorm".

    process_error: pd.Series
        Reserve estimate process error indexed by origin. Represents the
        risk associated with the projection of future contingencies that
        are inherently variable, even if parameters are known with certainty.

    parameter_error: pd.Series
        Reserve estimate parameter error indexed by origin. Represents
        the risk that the parameters used in the methods or models are not
        representative of future outcomes.

    devpvar: pd.Series
        The development period variance, usually represented as
        :math:`\\hat{\\sigma}^{2}_{k}` in the literature. For a triangle having
        n development periods, ``devpvar`` will contain n-1
        elements.

    ldfvar: pd.Series
        Variance of age-to-age factors. Required for Murphy's recursive
        estimator of parameter risk. For a triangle having n
        development periods, ``ldfvar`` will contain n-1 elements.

    rvs: pd.Series
        Series indexed by origin containing Scipy frozen random variable
        with parameters mu and sigma having distribution specified by
        ``dist``.

    kwargs: dict
        Additional parameters originally passed into ``MackChainLadder``'s
        ``__call__`` method.
    """
    def __init__(self, summary, tri, alpha, tail, ldfs, trisqrd, dist, process_error,
                 parameter_error, devpvar, ldfvar, rvs, **kwargs):

        super().__init__(summary=summary, tri=tri, ldfs=ldfs, tail=tail, trisqrd=trisqrd,
                         process_error=process_error, parameter_error=parameter_error)

        self.alpha = alpha
        self.dist = dist
        self.devpvar = devpvar
        self.ldfvar = ldfvar
        self.rvs = rvs

        if kwargs is not None:
            for kk in kwargs:
                setattr(self, kk, kwargs[kk])

        # Add formats for method-specific columns.
        mack_summ_hdrs = {ii: "{:,.0f}".format for ii in self.summary.columns if ii.endswith("%")}
        self._summspecs.update(mack_summ_hdrs)


    def _residuals_by_devp(self):
        """
        Calculate standardized residuals by development period.

        Returns
        -------
        pd.DataFrame
        """
        devpx = self.tri.devp.values[:-1]
        devpy = np.roll(devpx, shift=-1)
        xy = list(zip(devpx, devpy))[:-1]
        resids_list = []
        for x, y in xy:
            # Multiply triangle values at development period x by ldf at development
            # period x, then compute difference against triangle values at development
            # period y.
            yact_init = self.tri[y]
            comp_indices = yact_init[~pd.isnull(yact_init)].index
            n = comp_indices.size
            yact = yact_init[comp_indices]
            yhat = self.tri.loc[comp_indices, x] * self.ldfs.get(x)
            dfresid = pd.Series(yhat - yact, name="residuals").to_frame()
            dfresid["sigma"] = np.sqrt((dfresid["residuals"]**2).sum() / n - 1)
            dfresid["std_residuals"] = dfresid["residuals"] / (dfresid["sigma"] * np.sqrt((n - 1) / n))
            dfresid["t"] = x
            resids_list.append(dfresid)
        return(pd.concat(resids_list))


    def _residuals_by_origin(self):
        """
        Calculate standardized residuals by origin period.

        Returns
        -------
        pd.DataFrame
        """
        resids_list = []
        for origin in self.tri.origins[:-1]:
            dfresid = self.tri.loc[origin].to_frame().rename({origin: "yact"}, axis=1)
            dfresid["yhat"] = (dfresid["yact"] * self.ldfs).shift(1)
            dfresid = dfresid.dropna(how="any")
            dfresid["residuals"] = dfresid["yhat"] - dfresid["yact"]
            n = dfresid.shape[0]
            dfresid["sigma"] = np.sqrt((dfresid["residuals"]**2).sum() / n - 1)
            dfresid["std_residuals"] = dfresid["residuals"] / (dfresid["sigma"] * np.sqrt((n - 1) / n))
            dfresid["t"] = origin
            resids_list.append(dfresid)
        return(pd.concat(resids_list))


    def _spearman_corr_coeffs(self):
        """
        Compute the Spearman correlation coefficients for each pair of equal
        sized columns from ``self.tri._ranked_a2a``.

        For adjacent columns, a Spearman coefficient close to 0 indicates that
        the development factors between development years k - 1 and k and
        those between developmenr years k and k+1 are uncorrelated. Any
        other value of :math:`T_{k}` indicates that the factors are positively or
        negatively correlated.

        In the resulting DataFrame, columns are defined as:

            - k:
                An enumeration of the target development period.

            - w:
                Quanity used to weight the Spearman coefficient, specified
                as n - k - 1, where n is the number of origin periods
                in the triangle.

            - T_k:
                Spearman correlation coefficient. Defined as :math:`T_{k} = 1 - 6 \\sum_{i=1}^{n-k}`.

        Returns
        -------
        pd.DataFrame

        References
        ----------
        1. Mack, Thomas (1993) *Measuring the Variability of Chain Ladder Reserve
        Estimates*, 1993 CAS Prize Paper Competition on Variability of Loss Reserves.
        """
        dfranks = self.tri.ranked_a2a
        coeffs_list = []
        s_indices = [2 * ii for ii in range(dfranks.columns.size // 2)]
        r_indices = [1 + ii for ii in s_indices]
        n = self.tri.devp.size
        for s_indx, r_indx in zip(s_indices, r_indices):
            # s_indx, r_indx = 0, 1
            k = int(dfranks.iloc[:, s_indx].name.replace("s_", ""))
            w = n - k - 1
            s_ii = dfranks.iloc[:, s_indx]
            r_ii = dfranks.iloc[:, r_indx]
            T_k = 1 - 6 * np.sum((r_ii - s_ii)**2 / ((n - k)**3 - n + k))
            coeffs_list.append({"k": k, "w": w, "T_k": T_k})
        return(pd.DataFrame().from_dict(coeffs_list, orient="columns"))


    def _spearman_corr_total(self):
        """
        Weighted average of each adjacent column's Spearman coefficient
        from ``self._spearman_corr_coeffs``. Correlation coefficients are weighted
        by

        """
        # Compute w-weighted average of T_k from _spearman_corr_coeffs.
        df = self._spearman_corr_coeffs()
        return((df["w"] * df["T_k"]).sum() / df["w"].sum())


    def _devp_corr_test_var(self):
        """
        Return the variance used in the development period correlation test.

        Returns
        -------
        float
        """
        return((self.tri.devp.size - 2) * (self.tri.devp.size - 3) / 2)


    def devp_corr_test(self, p=.50):
        """
        Significance test to assess the degree of development period correlation.
        The first element of the returned tuple contains the upper and lower
        bounds of the test interval. The second element represents the test
        statistic, the weighted average of Spearman rank correlation coefficients.
        If the test statistic falls within the range bounded by the first element,
        the null hypothesis of having uncorrelated development factors is not
        rejected. If the test statistic falls outside the interval, development
        period correlations should be analyzed in greater detail.

        Parameters
        ----------
        p: float
            Represents the central normal interval outside of which
            development factors are assumed to exhibit some degree of
            correlation.

        Returns
        -------
        tuple
        """
        def fnorm(x, p):
            return(norm.cdf(x) - norm.cdf(-x) - p)

        f = functools.partial(fnorm, p=p)
        z = root(f, np.random.rand()).x.item()
        s = np.sqrt(self._devp_corr_test_var())
        return((-z / s, z / s), self._spearman_corr_total())



    def _cy_effects_table(self):
        """
        Construct a tabular summary of values used in assessing the presence of
        significant calendar year influences in the set of age-to-age factors.
        Resulting DataFrame contains the following columns:

        - j:
            The diagonal in question. For a triangle with n periods, j ranges
            from 2 to n - 1. The most recent diagonal is associated with
            j = n - 1.

        - S:
            Represents the number of small age-to-age factors for a given
            diagonal. Recall that small age-to-age factors are those less
            than the median for a given development period.

        - L:
            Represents the number of large age-to-age factors for a given
            diagonal. Recall that large age-to-age factors are those greater
            than the median for a given development period.

        - Z:
            For a given j, :math:`Z = min(S, L)`.

        - n:
            For a given j, is defined as :math:`S + L`.

        - m:
            For a given j, is defined as :math:`floor([n - 1] / 2)`.

        - E_Z:
            For a given j, is defined as :math:`\\frac{n}{2} - \\binom{n-1}{m} \\times \\frac{n}{2^{n}}`.

        - V_Z:
            For a given j, is defined as
            :math:`\\frac{n(n-1)}{4} - \\binom{n-1}{m} \\times \\frac{n(n-1)}{2^{n}} + \\mu - \\mu^{2}`.

        Returns
        -------
        pd.DataFrame
        """
        summ_list = []
        dflvi = self.tri.a2a_lvi
        dfhl = self.tri.a2a_assignment
        dflvi = dflvi.reset_index(drop=False).rename({"index": "dev"}, axis=1)
        devpinc = dflvi["dev"].diff(1).dropna().unique().item()
        origininc = dflvi["origin"].diff(-1).dropna().unique().item()
        diag_indx = sorted(list(enumerate(dflvi["dev"].values[::-1], start=1)), key=lambda v: v[-1])
        dfhl2 = dfhl.reset_index(drop=False).rename({"index": "origin"}, axis=1).melt(
            id_vars=["origin"], var_name=["dev"]).dropna(subset=["value"])

        # Begin with latest diagonal. Work backwards.
        for jj, devp in diag_indx[:-1]:
            dcounts = dflvi.merge(dfhl2, on=["origin", "dev"])["value"].value_counts()
            dsumm = {"j": jj, "S": dcounts.get(-1, 0), "L": dcounts.get(1, 0)}
            summ_list.append(dsumm)
            # Adjust dflvi by shifting origin period back a single period
            # relative to devp. Drop record in which origin is missing.
            dflvi["origin"] = dflvi["origin"].shift(-1)
            dflvi = dflvi[~pd.isnull(dflvi.origin)]
            dflvi["origin"] = dflvi["origin"].astype(int)

        dfcy = pd.DataFrame().from_dict(summ_list).sort_values("j")

        # Z is defined as the minimum of S and L by record.
        dfcy["Z"] = dfcy.apply(lambda rec: min(rec.S, rec.L), axis=1)

        # n is defined as the sum of S + L by record.
        dfcy["n"] = dfcy["S"] + dfcy["L"]

        # m is defined as the largest integer <= (n - 1) / 2.
        dfcy["m"] = np.floor(.50 * (dfcy["n"] - 1))

        # E_Z is the expected value of Z.
        dfcy["E_Z"] = .50 * dfcy["n"] - special.binom(dfcy["n"].values - 1, dfcy["m"].values) * \
            (dfcy["n"] / np.power(2, dfcy["n"]))

        # V_Z is the variance of Z.
        dfcy["V_Z"] = .25 * dfcy["n"] * (dfcy["n"] - 1) - \
            special.binom(dfcy["n"].values - 1, dfcy["m"].values) * \
            (dfcy["n"] * (dfcy["n"] - 1) / np.power(2, dfcy["n"])) + \
            dfcy["E_Z"] - dfcy["E_Z"]**2

        return(dfcy)


    def cy_effects_test(self, p=.05):
        """
        We reject the hypothesis, with an error probability of p, of having no
        significant calendar year effects only if not::

            E_Z - 1.96 * sqrt(V_Z)  <=  Z  <=  E_Z + 1.96 * sqrt(V_Z) (if p = .05)

        Parameters
        ----------
        p: float
            Significance level with which to perform test for calendar year
            effects. Test is two-sided (see above).

        Returns
        -------
        tuple
        """
        dfcy = self._cy_effects_table().sum()
        Z = dfcy.get("Z")
        mu = dfcy.get("E_Z")
        v = dfcy.get("V_Z")
        var_mult = np.abs(norm.ppf(p / 2))
        lb = mu - var_mult * np.sqrt(v)
        ub = mu + var_mult * np.sqrt(v)
        return((lb, ub), Z)


    def _mack_data_transform(self):
        """
        Generate data by origin period and in total to plot estimated reserve distributions.

        Returns
        -------
        pd.DataFrame
        """
        df = pd.concat([
            pd.DataFrame({
                "origin": ii, "x": np.linspace(self.rvs[ii].ppf(.001), self.rvs[ii].ppf(.999), 100),
                "y": self.rvs[ii].pdf(np.linspace(self.rvs[ii].ppf(.001), self.rvs[ii].ppf(.999), 100))
                })
            for ii in self.rvs.index
            ])

        return(df.dropna(how="all", subset=["x", "y"]))


    def get_quantiles(self, q):
        """
        Get quantiles of estimated reserve distribution for an individual origin periods and
        in total. Returns a DataFrame, with columns representing the percentiles of interest.

        Parameters
        ----------
        q: array_like of float or float
            Quantile or sequence of quantiles to compute, which must be between 0 and 1
            inclusive.

        Returns
        -------
        pd.DataFrame
        """
        qarr = np.asarray(q, dtype=np.float)
        if np.any(np.logical_and(qarr > 1, qarr < 0)):
            raise ValueError("q values must fall within [0, 1].")
        else:
            qtls, qtlhdrs = self._qtls_formatter(q=q)
            qtl_pairs = [(qtlhdrs[ii], qtls[ii]) for ii in range(len(qtls))]
            dqq = {
                str(ii[0]): [self.rvs[origin].ppf(ii[-1]) for origin in self.rvs.index]
                    for ii in qtl_pairs
                }
        return(pd.DataFrame().from_dict(dqq).set_index(self.summary.index))



    def plot(self, q=.95, dist_color="#000000", q_color="#E02C70", axes_style="darkgrid",
             context="notebook", col_wrap=4, exhibit_path=None):
        """
        Plot estimated reserve distribution by origin year and in total.
        The mean of the reserve estimate will be highlighted, along with
        and quantiles specified in ``q``.

        Parameters
        ----------
        q: float in range of [0,1]
            The quantile to highlight, which must be between 0 and 1 inclusive.

        dist_color: str
            Color or hexidecimal color code of estimated reserve distribution.

        q_color: str
            Color or hexidecimal color code of estimated reserve mean
            and quantiles.

        axes_style: {"darkgrid", "whitegrid", "dark", "white", "ticks"}
            Aesthetic style of seaborn plots. Default values is "darkgrid".

        context: {"notebook", "paper", "talk", "poster"}.
            Set the plotting context parameters. According to the seaborn
            documentation, This affects things like the size of the labels,
            lines, and other elements of the plot, but not the overall style.
            Default value is "notebook".

        col_wrap: int
            The maximum number of origin period axes to have on a single row
            of the resulting FacetGrid. Defaults to 5.

        exhibit_path: str
            Path to which exhibit should be written. If None, exhibit will be
            rendered via ``plt.show()``.

        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_context(context)

        data = self._data_transform()

        with sns.axes_style(axes_style):
            qtls, qtlhdrs = self._qtls_formatter(q)
            grid = sns.FacetGrid(
                data, col="origin", col_wrap=col_wrap, margin_titles=False,
                despine=True, sharex=False, sharey=False
                )

            grid.set_axis_labels("", "")
            grid.map(plt.plot, "x", "y", color=dist_color, linewidth=1.25)

            for origin, ax_ii in zip(self.rvs.index[1:], grid.axes):
                # Determine reserve estimate at mean and specified quantiles.
                rv_ii = self.rvs[origin]
                with np.errstate(invalid="ignore"):
                    qq_ii = [rv_ii.ppf(jj) for jj in qtls]
                vline_pos, vline_str = [rv_ii.mean()] + qq_ii, ["mean"] + qtlhdrs

                for ii, jj in zip(vline_pos, vline_str):
                    # Draw vertical line and annotation.
                    ax_ii.axvline(
                        x=ii, linewidth=1., linestyle="--", color=q_color
                        )
                    ax_ii.annotate(
                        "{} = {:,.0f}".format(jj, ii), xy=(ii, 0), xytext=(3.5, 0),
                        textcoords="offset points", fontsize=8, rotation=90,
                        color="#000000",
                        )
                    ax_ii.annotate(
                        origin, xy=(.85, .925), xytext=(.85, .925), xycoords='axes fraction',
                        textcoords='axes fraction', fontsize=9, rotation=0, color="#000000",
                        )
                    ax_ii.set_xticklabels([])
                    ax_ii.set_yticklabels([])
                    ax_ii.set_title("")
                    ax_ii.set_xlabel("")
                    ax_ii.set_ylabel("")
                    ax_ii.grid(True)

                # Draw border around each facet.
                for _, spine in ax_ii.spines.items():
                    spine.set(visible=True, color="#000000", linewidth=.50)

            if exhibit_path is not None:
                plt.savefig(exhibit_path)
            else:
                plt.show()


    def diagnostics(self, **kwargs):
        """
        Statistical diagnostics plots of Mack Chain Ladder estimator.
        Exhibit is a faceted quad plot, representing the estimated
        reserve distribution, the path to ultimate for each origin period,
        and residuals by origin and development period.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        pltkwargs = dict(
            marker="s", markersize=5, alpha=1, linestyle="-", linewidth=1.,
            figsize=(9, 9), cmap="rainbow",
            )
        if kwargs:
            pltkwargs.update(kwargs)

        dfdens_all = self._mack_data_transform()
        dfdens = dfdens_all[(dfdens_all.origin == "total")]
        dfresid0 = self._residuals_by_devp()
        dfresid1 = self._residuals_by_origin()
        cl_data = pd.melt(self.trisqrd, var_name="dev", ignore_index=False).reset_index(
            drop=False).rename({"index": "origin"}, axis=1)
        grps = cl_data.groupby("origin", as_index=False)
        data_list = [grps.get_group(ii) for ii in self.tri.origins]

        # Get unique hex color for each unique origin period.
        fcolors = cm.get_cmap(pltkwargs["cmap"], len(self.tri.origins))
        colors_rgba = [fcolors(ii) for ii in np.linspace(0, 1, len(self.tri.origins))]
        colors_hex = [mpl.colors.to_hex(ii, keep_alpha=False) for ii in colors_rgba]

        fig, ax = plt.subplots(2, 2, figsize=pltkwargs["figsize"], tight_layout=True)

        # Upper left facet represents aggregate reserve distribution.
        mean_agg_reserve = self.summary.at["total", "reserve"]
        ax[0, 0].set_title("Aggregate Reserve Distribution", fontsize=8, loc="left", weight="bold")
        ax[0, 0].plot(dfdens.x.values, dfdens.y.values, color="#000000", linewidth=1.)
        ax[0, 0].axvline(mean_agg_reserve, linestyle="--", color="red", linewidth=1.)
        ax[0, 0].get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: format(int(v), ",")))
        ax[0, 0].set_ylim(bottom=0)
        ax[0, 0].set_xlim(left=0)
        ax[0, 0].set_xlabel("")
        ax[0, 0].set_ylabel("")
        ax[0, 0].tick_params(axis="x", which="major", direction="in", labelsize=6)
        ax[0, 0].set_yticks([])
        ax[0, 0].set_yticklabels([])
        ax[0, 0].xaxis.set_ticks_position("none")
        ax[0, 0].yaxis.set_ticks_position("none")
        ax[0, 0].grid(False)
        ax[0, 0].annotate(
            "mean total reserve = {:,.0f}".format(mean_agg_reserve),
            xy=(mean_agg_reserve, self.rvs.total.pdf(mean_agg_reserve) * .50),
            xytext=(-9, 0), textcoords="offset points", fontsize=8, rotation=90,
            color="#000000",
            )
        for axis in ["top", "bottom", "left", "right"]:
            ax[0, 0].spines[axis].set_linewidth(0.5)


        # Upper right facet represents Chain Ladder development paths (trisqrd).
        for ii, hex_color, dforg in zip(range(len(colors_hex)), colors_hex, data_list):
            xx = dforg["devp"].values
            yy = dforg["value"].values
            marker = self._markers[ii % len(self._markers)]
            yy_divisor = 1  # 1000 if np.all(yy>1000) else 1
            yy_axis_label = "(000's)" if yy_divisor == 1000 else ""
            ax[0, 1].plot(
                xx, yy / yy_divisor, color=hex_color,
                linewidth=pltkwargs["linewidth"], linestyle=pltkwargs["linestyle"],
                label=dforg.origin.values[0], marker=marker,
                markersize=pltkwargs["markersize"]
                )
        for axis in ["top", "bottom", "left", "right"]:
            ax[0, 1].spines[axis].set_linewidth(0.5)

        ax[0, 1].set_title("Development by Origin", fontsize=8, loc="left", weight="bold")
        ax[0, 1].get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: format(int(v), ",")))
        ax[0, 1].set_xlabel("dev", fontsize=8)
        ax[0, 1].set_ylabel(yy_axis_label, fontsize=7)
        ax[0, 1].set_ylim(bottom=0)
        ax[0, 1].set_xlim(left=cl_data["devp"].min())
        ax[0, 1].set_xticks(np.sort(cl_data["devp"].unique()))
        ax[0, 1].tick_params(axis="x", which="major", direction="in", labelsize=6)
        ax[0, 1].tick_params(axis="y", which="major", direction="in", labelsize=6)
        ax[0, 1].xaxis.set_ticks_position("none")
        ax[0, 1].yaxis.set_ticks_position("none")
        ax[0, 1].grid(False)
        ax[0, 1].legend(loc="lower right", fancybox=True, framealpha=1, fontsize="x-small")


        # Lower left facet represents residuals by development period.
        ax[1, 0].set_title("std. residuals by devp", fontsize=8, loc="left", weight="bold")
        ax[1, 0].scatter(
            dfresid0["t"].values, dfresid0["std_residuals"].values,
            marker="o", edgecolor="#000000", color="#FFFFFF", s=15,
            linewidths=.75
            )
        ax[1, 0].axhline(0, linestyle="dashed", linewidth=1.)
        ax[1, 0].set_xlabel("dev", fontsize=7)
        ax[1, 0].set_ylabel("std. residuals", fontsize=7)
        ax[1, 0].set_xticks(np.sort(dfresid0.t.unique()))
        ax[1, 0].tick_params(axis="x", which="major", direction="in", labelsize=6)
        ax[1, 0].tick_params(axis="y", which="major", direction="in", labelsize=6)
        ax[1, 0].xaxis.set_ticks_position("none")
        ax[1, 0].yaxis.set_ticks_position("none")
        ax[1, 0].grid(False)
        for axis in ["top", "bottom", "left", "right"]:
            ax[1, 0].spines[axis].set_linewidth(0.5)

        # Lower right facet represents residuals by origin period.
        ax[1, 1].set_title("std. residuals by origin", fontsize=8, loc="left", weight="bold")
        ax[1, 1].scatter(
            dfresid1["t"].values, dfresid1["std_residuals"].values,
            marker="o", edgecolor="#000000", color="#FFFFFF", s=15,
            linewidths=.75
            )
        ax[1, 1].axhline(0, linestyle="dashed", linewidth=1.)
        ax[1, 1].set_xlabel("origin", fontsize=8)
        ax[1, 1].set_ylabel("std. residuals", fontsize=8)
        ax[1, 1].set_xticks(np.sort(dfresid1.t.unique()))
        ax[1, 1].tick_params(axis="x", which="major", direction="in", labelsize=6)
        ax[1, 1].tick_params(axis="y", which="major", direction="in", labelsize=6)
        ax[1, 1].xaxis.set_ticks_position("none")
        ax[1, 1].yaxis.set_ticks_position("none")
        ax[1, 1].grid(False)
        for axis in ["top", "bottom", "left", "right"]:
            ax[1, 1].spines[axis].set_linewidth(0.5)

        plt.show()
