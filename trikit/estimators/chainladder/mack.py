"""
_MackChainLadder implementation.
"""
import functools
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from . import BaseChainLadder, BaseChainLadderResult


class MackChainLadder(BaseChainLadder):
    """
    Mack Chain Ladder estimator. The predicition variance is comprised
    of the estimation variance and the process variance. Estimation variance
    arises from the inability to accurately define the distribution from which
    past events have been generated. Process variance arises from the
    inability to accurately predict which single outcome from the distribution
    will occur at a given time. The predicition error is defined as the
    standard deviation of the forecast.

    References
    ----------
    1. Mack, Thomas (1993) *Measuring the Variability of Chain Ladder Reserve
       Estimates*, 1993 CAS Prize Paper Competition on 'Variability of Loss Reserves'.

    2. Mack, Thomas, (1993), *Distribution-Free Calculation of the Standard Error
       of Chain Ladder Reserve Estimates*, ASTIN Bulletin 23, no. 2:213-225.

    3. Mack, Thomas, (1999), *The Standard Error of Chain Ladder Reserve Estimates:
       Recursive Calculation and Inclusion of a Tail Factor*, ASTIN Bulletin 29,
       no. 2:361-366.

    4. England, P., and R. Verrall, (2002), *Stochastic Claims Reserving in General
      Insurance*, British Actuarial Journal 8(3): 443-518.

    5. Murphy, Daniel, (2007), *Chain Ladder Reserve Risk Estimators*, CAS E-Forum,
       Summer 2007.

    6. Carrato, A., McGuire, G. and Scarth, R. 2016. *A Practitioner's
       Introduction to Stochastic Reserving*, The Institute and Faculty of
       Actuaries. 2016.
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



    def __call__(self, alpha=1, tail=1.0, dist="lognorm", q=[.75, .95], two_sided=False):
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
            self._process_error(ldfs=ldfs, devpvar=devpvar).iloc[:,-1].replace(np.NaN, 0),
            name="process_error", dtype=np.float
            )
        param_error = pd.Series(
            self._parameter_error(ldfs=ldfs, ldfvar=ldfvar).iloc[:,-1].replace(np.NaN, 0),
            name="parameter_error", dtype=np.float
            )
        mse = self._mean_squared_error(
            process_error=proc_error, parameter_error=param_error
            )
        std_error = pd.Series(np.sqrt(mse), name="std_error")
        cv = pd.Series(std_error / reserves, name="cv")
        trisqrd = self._trisqrd(ldfs=ldfs)

        dfmatur = maturity.to_frame().reset_index(drop=False).rename({"index":"origin"}, axis=1)
        dfcldfs = cldfs.to_frame().reset_index(drop=False).rename({"index":"maturity"}, axis=1)
        dfcldfs["maturity"] = dfcldfs["maturity"].astype(np.str)
        dfcldfs["emergence"] = 1 / dfcldfs["cldf"]
        dfsumm = dfmatur.merge(dfcldfs, on=["maturity"], how="left").set_index("origin")
        dfsumm.index.name = None
        dflatest = latest.to_frame().rename({"latest_by_origin":"latest"}, axis=1)
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
        n = self.tri.devp.size

        # Compute mse for aggregate reserve.
        mse_total = pd.Series(index=dfsumm.index[:-1], dtype=np.float)
        quotient = pd.Series(devpvar / ldfs**2, dtype=np.float).reset_index(drop=True)
        quotient.index = quotient.index + 1

        for indx, ii in enumerate(mse_total.index[1:], start=2):
            mse_ii, ult_ii = mse[ii],  ultimates[ii]
            ults_sum = ultimates[ultimates.index>ii].dropna().sum()
            rh_sum = sum(
                quotient[jj] / sum(trisqrd.loc[mm, jj] for mm in range(1, (n - jj) + 1))
                for jj in range(n + 1 - indx, n)
                )
            mse_total[ii] = mse_ii + 2 * ult_ii * ults_sum * rh_sum


        # Reset trisqrd columns and index back to original values.
        trisqrd.columns, trisqrd.index = self.tri.columns, self.tri.index
        dfsumm.loc["total", "std_error"] = np.sqrt(mse_total.dropna().sum())
        dfsumm.loc["total", "cv"] = dfsumm.loc["total", "std_error"] / dfsumm.loc["total", "reserve"]

        if dist=="norm":
            std_params, mean_params = dfsumm["std_error"], dfsumm["reserve"]
            rv_list = [norm(loc=ii, scale=jj) for ii,jj in zip(mean_params, std_params)]

        elif dist=="lognorm":
            with np.errstate(divide="ignore"):
                std_params = np.sqrt(np.log(1 + (dfsumm["std_error"] / dfsumm["reserve"])**2)).replace(np.NaN, 0)
                mean_params = np.clip(np.log(dfsumm["reserve"]), a_min=0, a_max=None) - .50 * std_params**2
                rv_list = [lognorm(scale=np.exp(ii), s=jj) for ii,jj in zip(mean_params, std_params)]

        else:
            raise ValueError(
                "dist must be one of {'norm', 'lognorm'}, not `{}`.".format(dist)
                )

        rvs = pd.Series(rv_list, index=dfsumm.index)
        qtls, qtlhdrs = self._qtls_formatter(q=q, two_sided=two_sided)

        # Populate qtlhdrs columns with estimated quantile estimates.
        with np.errstate(invalid="ignore"):
            for ii, jj in zip(qtls, qtlhdrs):
                for origin in rvs.index:
                    dfsumm.loc[origin, jj] = rvs[origin].ppf(ii)

        dfsumm.loc[self.tri.index.min(), ["cv"] + qtlhdrs] = np.NaN

        # Instantiate and return MackChainLadderResult instance.
        kwds = {
            "alpha":alpha, "tail":tail, "dist":dist, "q":q, "two_sided":two_sided,
            }

        mcl_result = MackChainLadderResult(
            summary=dfsumm, tri=self.tri, ldfs=ldfs, trisqrd=trisqrd,
            process_error=proc_error, parameter_error=param_error, devpvar=devpvar,
            ldfvar=ldfvar, mse=mse, mse_total=mse_total, cv=cv, rvs=rvs, **kwds
            )

        return(mcl_result)




    @staticmethod
    def _qtls_formatter(q, two_sided=False):
        """
        Return array_like of formatted quantiles for MackChainLadder
        summary.

        Parameters
        ----------
        q: array_like of float or float
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
        tuple of list
        """
        qtls = np.asarray([q] if isinstance(q, (float, int)) else q)

        if np.all(np.logical_and(qtls <= 1, qtls >= 0)):
            if two_sided:
                qtls = np.sort(np.unique(np.append((1 - qtls) / 2., (1 + qtls) / 2.)))
            else:
                qtls = np.sort(np.unique(qtls))
        else:
            raise ValueError("Values for quantiles must fall between [0, 1].")

        qtlhdrs = [
            "{:.5f}".format(ii).rstrip("0").rstrip(".") + "%" for ii in 100 * qtls
            ]

        return(qtls, qtlhdrs)



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
        ldfs = (self.tri.a2a * w * C**alpha).sum(axis=0) / (w * C**alpha).sum(axis=0)
        increment = np.unique(ldfs.index[1:] - ldfs.index[:-1])[0]
        ldfs.loc[ldfs.index.max() + increment] = tail
        return(ldfs)


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
            Reserve estimate process error indexed by origin. Represents the
            risk associated with the projection of future contingencies that
            are inherently variable, even if/when the parameters are known
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














class MackChainLadderResult(BaseChainLadderResult):
    """
    MackChainLadder output.
    """
    def __init__(self, summary, tri, ldfs, trisqrd, process_error, parameter_error,
                 devpvar, ldfvar, mse, mse_total, cv, rvs, **kwargs):
        """
        Container class for ``MackChainLadder`` output.

        Parameters
        ----------
        summary: pd.DataFrame
            ``MackChainLadder`` summary.

        tri: trikit.triangle.CumTriangle
            A cumulative triangle instance.

        ldfs: pd.Series
            Loss development factors.

        process_error: pd.Series
            Reserve estimate process error indexed by origin. Represents the
            risk associated with the projection of future contingencies that
            are inherently variable, even if/when the parameters are known
            with certainty.

        parameter_error: pd.Series
            Reserve estimate parameter error indexed by origin. Represents
            the risk that the parameters used in the methods or models are not
            representative of future outcomes.

        mse: pd.Series
            Estimated mean squared error (mse) of each origin period. The
            earliest origin period's mse will be 0, since it is assumed to be
            fully developed.

        mse_total: pd.Series
            Mean squared error (mse) of aggregate reserve estimate. Reserve
            estimators for individual origin periods are correlated due to the
            fact that they rely on the same parameters $\hat{f}_{j}$ and
            $\hat{\sigma}^2_{j}$ (``ldfs`` and ``devpvar`` respectively). To
            compute the mean square error for the total reserve, we aggregate
            mean square error estimates for individual origin periods
            (available in ``mse`` parameter), plus an allowance for the
            correlation between estimators[6]. The values in ``mse_total`` are
            aggregated to obtain an estimate for the mean squared error of the
            total reserve.

        devpvar: pd.Series
            The development period variance, usually represented as
            $\hat{\sigma}^{2}_{k}$ in the literature. For a triangle having
            ``n`` development periods, ``devpvar`` will contain ``n-1``
            elements.

        ldfvar: pd.Series
            Variance of age-to-age factors. Required for Murphy's recursive
            estimator of parameter risk. For a triangle having ``n``
            development periods, ``ldfvar`` will contain ``n-1`` elements.

        rvs: pd.Series
            Series indexed by origin containing Scipy frozen random variable
            with parameters mu and sigma having distribution specified by
            ``dist``.

        cv: pd.Series.
            Coefficient of variation, the ratio of standard deviation and mean.
             Here, ``cv = std_error / reserves``.

        kwargs: dict
            Additional parameters originally passed into ``MackChainLadder``'s
            ``__call__`` method.
        """
        super().__init__(summary=summary, tri=tri, ldfs=ldfs, trisqrd=trisqrd, **kwargs)

        self.std_error = summary["std_error"]
        self.cv = summary["cv"]
        self.parameter_error = parameter_error
        self.process_error = process_error
        self.mse_total = mse_total
        self.summary = summary
        self.devpvar = devpvar
        self.trisqrd = trisqrd
        self.ldfvar = ldfvar
        self.ldfs = ldfs
        self.tri = tri
        self.mse = mse
        self.rvs = rvs

        if kwargs is not None:
            for kk in kwargs:
                setattr(self, kk, kwargs[kk])

        # Add formats for method-specific columns.
        mack_summ_hdrs = {ii:"{:,.0f}".format for ii in self.summary.columns if ii.endswith("%")}
        mack_summ_hdrs.update({"std_error":"{:,.0f}".format, "cv":"{:.5f}".format})
        self._summspecs.update(mack_summ_hdrs)

        # Quantile suffix for plot method annotations.
        self.dsuffix = {
            "0":"th", "1":"st", "2":"nd", "3":"rd", "4":"th", "5":"th", "6":"th",
            "7":"th", "8":"th", "9":"th",
            }


    def _qtls_formatter(self, q):
        """
        Return array_like of actual and formatted quantiles for MackChainLadder
        summary.

        Parameters
        ----------
        q: array_like of float or float
            Quantile or sequence of quantiles to compute, which must be
            between 0 and 1 inclusive.

        Returns
        -------
        tuple of list
        """
        qtls = np.asarray([q] if isinstance(q, (float, int)) else q)
        if np.all(np.logical_and(qtls <= 1, qtls >= 0)):
            qtls = np.sort(np.unique(qtls))
        else:
            raise ValueError("Values for quantiles must fall between [0, 1].")

        qtlhdrs = [
            "{:.5f}".format(ii).rstrip("0").rstrip(".") for ii in 100 * qtls
            ]
        qtlhdrs = [
            ii + "th" if "." in ii else ii + self.dsuffix[ii[-1]] for ii in qtlhdrs
            ]
        return(qtls, qtlhdrs)


    def _data_transform(self):
        """
        Generate data by origin period and in total to plot estimated
        reserve distributions.

        Returns
        -------
        pd.DataFrame
        """
        df = pd.concat([
            pd.DataFrame({
                "origin":ii, "x":np.linspace(self.rvs[ii].ppf(.001), self.rvs[ii].ppf(.999), 100),
                "y":self.rvs[ii].pdf(np.linspace(self.rvs[ii].ppf(.001), self.rvs[ii].ppf(.999), 100))
                })
            for ii in self.rvs.index
            ])

        return(df.dropna(how="all", subset=["x", "y"]))



    def get_quantiles(self, q=[.05, .25, .50, .75, .95], origin=None):
        """
        Get quantile of estimated reserve distribution for an individual
        origin period or in total. Returns a tuple of ndarrays: The first
        representing user-provided quantiles, the second representing values
        estimated by the reserve distribution.

        Parameters
        ----------
        q: array_like of float or float
            Quantile or sequence of quantiles to compute, which must be
            between 0 and 1 inclusive. Default value is
            ``[.05, .25, .50, .75, .95]``.

        origin: int
            Target origin period from which to return specified quantile(s).
            If None, quantiles from aggregate reserve distribution will
            be returned.

        Returns
        -------
        tuple of ndarrays
        """
        pass


    def plot(self, q=.95, dist_color="#000000", q_color="#E02C70", axes_style="darkgrid",
             context="notebook", col_wrap=4, exhibit_path=None):
        """
        Plot estimated reserve distribution by origin year and in total.
        The mean of the reserve estimate will be highlighted, along with
        and quantiles specified in ``q``.

        Parameters
        ----------
        q: float in range of [0,1]
            Two-sided percentile interval to highlight, which must be between
            0 and 1 inclusive. For example, when ``q=.90``, the 5th and
            95th percentile of the reserve distribution will is highlighted
            in each facet $(\frac{1 - q}{2}, \frac(1 + q}{2})$.

        dist_color: str
            Color or hexidecimal color code of estimated reserve distribution.

        q_color: str
            Color or hexidecimal color code of estimated reserve mean
            and quantiles.

        axes_style: {"darkgrid", "whitegrid", "dark", "white", "ticks"}
            Aesthetic style of seaborn plots. Default values is "darkgrid".

        context: {"paper", "talk", "poster"}.
            Set the plotting context parameters. According to the seaborn
            documentation, This affects things like the size of the labels,
            lines, and other elements of the plot, but not the overall style.
            Default value is ``"notebook"``.

        col_wrap: int
            The maximum number of origin period axes to have on a single row
            of the resulting FacetGrid. Defaults to 5.

        exhibit_path: str
            Path to which exhibit should be written. If None, exhibit will be
            rendered via ``plt.show()``.

        """
        import matplotlib
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

                for ii,jj in zip(vline_pos, vline_str):
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
                        textcoords='axes fraction', fontsize=10, rotation=0, color="#000000",
                        )
                    ax_ii.set_xticklabels([]); ax_ii.set_yticklabels([])
                    ax_ii.set_title(""); ax_ii.set_xlabel(""); ax_ii.set_ylabel("")
                    ax_ii.grid(True)

                # Draw border around each facet.
                for _, spine in ax_ii.spines.items():
                    spine.set(visible=True, color="#000000", linewidth=.50)

            if exhibit_path is not None:
                plt.savefig(exhibit_path)
            else:
                plt.show()


        def diagnostics(self):
            """
            Statistical diagnostics plots of Mack Chain Ladder estimator.
            """
            pass
