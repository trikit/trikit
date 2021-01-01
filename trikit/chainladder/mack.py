"""
_MackChainLadder implementation.
"""
import functools
import numpy as np
import pandas as pd
from scipy.stats import norm
from . import BaseChainLadder, BaseChainLadderResult



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
        # alpha = 1
        # tail = 1.0
        # dist = "lognorm"
        # q = [.75, .95]
        # two_sided = False


        ldfs = mcl._ldfs(alpha=alpha)
        cldfs = mcl._cldfs(ldfs=ldfs, tail=1.0)
        maturity = mcl.tri.maturity.astype(np.str)
        latest = mcl.tri.latest_by_origin
        ults = mcl._ultimates(cldfs=cldfs)
        ibnr = mcl._reserves(ultimates=ults)
        devpvar = mcl._devp_variance(ldfs=ldfs, alpha=alpha)
        ldfvar = mcl._ldf_variance(devpvar=devpvar, alpha=alpha)
        proc_error = pd.Series(
            mcl._process_error(ldfs=ldfs, devpvar=devpvar).iloc[:,-1].replace(np.NaN, 0),
            name="process_error", dtype=np.float
            )
        param_error = pd.Series(
            mcl._parameter_error(ldfs=ldfs, ldfvar=ldfvar).iloc[:,-1].replace(np.NaN, 0),
            name="parameter_error", dtype=np.float
            )
        mse = mcl._mean_squared_error(process_error=proc_error, parameter_error=param_error)
        std_error = pd.Series(np.sqrt(mse), name="std_error")
        cv = pd.Series(std_error / ibnr, name="cv")

        trisqrd_ldfs = ldfs.copy(deep=True)
        increment = np.unique(ldfs.index[1:] - ldfs.index[:-1])[0]
        trisqrd_ldfs.loc[ldfs.index.max() + increment] = tail
        trisqrd = mcl._trisqrd(ldfs=trisqrd_ldfs)

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

        # Add total index and set to NaN fields that shouldn't be aggregated.
        dfsumm.loc["total"] = dfsumm.sum()
        dfsumm.loc["total", "maturity"] = ""
        dfsumm.loc["total", "cldf"] = np.NaN


        # Create latest and trisqrd reference using 1-based indexing.
        latest = mcl.tri.latest.sort_index()
        latest.origin = range(1, latest.index.size + 1)
        latest.dev = range(latest.index.size, 0, -1)
        trisqrd = mcl._trisqrd(ldfs).drop("ultimate", axis=1)
        trisqrd.index = range(1, trisqrd.index.size + 1)
        trisqrd.columns = range(1, trisqrd.columns.size + 1)
        n = mcl.tri.devp.size

        # Compute mse for aggregate reserve.
        mse_total = pd.Series(index=dfsumm.index[:-1], dtype=np.float)
        quotient = pd.Series(devpvar / ldfs**2, dtype=np.float).reset_index(drop=True)
        quotient.index = quotient.index + 1

        for indx, ii in enumerate(mse_total.index[1:], start=2):
            mse_ii, ult_ii = mse[ii],  ults[ii]
            ults_sum = ults[ults.index>ii].dropna().sum()
            rh_sum = sum(
                quotient[jj] / sum(trisqrd.loc[mm, jj] for mm in range(1, (n - jj) + 1))
                for jj in range(n + 1 - indx, n)
                )
            mse_total[ii] = mse_ii + 2 * ult_ii * ults_sum * rh_sum

        dfsumm.loc["total", "std_error"] = np.sqrt(mse_total.dropna().sum())
        dfsumm.loc["total", "cv"] = dfsumm.loc["total", "std_error"] / dfsumm.loc["total", "reserve"]

        if dist=="norm":
            std_params, mean_params = dfsumm["std_error"], dfsumm["reserve"]

            def rv(mu, sigma, z):
                """
                Return evaluated Normal distribution with mean ``mu`` and
                standard deviation ``sigma`` at ``z``.
                """
                return(mu + sigma * z)

        if dist=="lognorm":
            std_params = np.sqrt(np.log(1 + (dfsumm["std_error"] / dfsumm["reserve"])**2)).replace(np.NaN, 0)
            mean_params = np.clip(np.log(dfsumm["reserve"]), a_min=0, a_max=None) - .50 * std_params**2

            def rv(mu, sigma, z):
                """
                Return evaluated Log-normal distribution with mean ``mu`` and
                standard deviation ``sigma`` at ``z``.
                """
                return(np.exp(mu + sigma * z))

        sigma = pd.Series(std_params, name="sigma", dtype=np.float)
        mu = pd.Series(mean_params, name="mu", dtype=np.float)
        dfsumm = dfsumm.join(mu.to_frame()).join(sigma.to_frame())

        # Determine quantiles.
        qtls = np.asarray([q] if isinstance(q, (float, int)) else q)
        if np.all(np.logical_and(qtls <= 1, qtls >= 0)):
            if two_sided:
                qtls = np.sort(np.unique(np.append((1 - qtls) / 2., (1 + qtls) / 2.)))
            else:
                qtls = np.sort(np.unique(qtls))
        else:
            raise ValueError("Values for quantiles must fall between [0, 1].")

        qtlsfmt = [
            "{:.5f}".format(i).rstrip("0").rstrip(".") + "%" for i in 100 * qtls
            ]
        for ii, jj in zip(qtls, qtlsfmt):
            dfsumm[jj] = dfsumm.apply(
                lambda rec: rv(rec.mu, rec.sigma, norm().ppf(ii)), axis=1
                )
        dfsumm.loc[mcl.tri.index.min(), ["cv"] + qtlsfmt] = np.NaN
        dfsumm = dfsumm.drop(["mu", "sigma"], axis=1)

        # Instantiate and return MackChainLadderResult instance.


        alpha = 1
        tail = 1.0
        dist = "lognorm"
        q = [.75, .95]
        two_sided = False


        mcl_result = MackChainLadderResult(
            summary=dfsumm,






            reserve_dist=dfreserves, sims_data=dfprocerror,
            tri=self.tri, ldfs=ldfs_, cldfs=cldfs_, latest=latest_,
            maturity=maturity_, ultimates=ultimates_, reserves=reserves_,
            scale_param=scale_param, unscaled_residuals=unscld_residuals_,
            adjusted_residuals=adjust_residuals_, sampling_dist=sampling_dist_res,
            fitted_tri_cum=tri_fit_cum_, fitted_tri_incr=tri_fit_incr_,
            trisqrd=trisqrd_, **kwds)



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














class MackChainLadderResult(BaseChainLadderResult):
    """
    MackChainLadder output.
    """
    def __init__(self, summary, tri, trisqrd, alpha, tail, dist, q, two_sided,
                 ldfs, cldfs, latest, maturity, ultimates, reserves, process_error,
                 parameter_error, devpvar, ldfvar, mse, mse_total, std_error, cv,
                 mu, sigma):

        self.summary = summary
        self.tri = tri
        self.trisqrd = trisqrd
        self.aplha = alpha
        self.tail = tail
        self.dist = dist
        self.q = q
        self.two_sided = two_sided
        self.ldfs = ldfs
        self.cldfs = cldfs
        self.latest = latest
        self.maturity = maturity
        self.ultimates = ultimates
        self.reserves = reserves
        self.process_error = process_error
        self.parameter_error = parameter_error
        self.devpvar = devpvar
        self.ldfvar = ldfvar
        self.mse = mse
        self.mse_total = mse_total
        self.std_error = std_error
        self.cv = cv
        self.mu = mu
        self.sigma = sigma




        """
        Container class for ``MackChainLadder``'s output.

        Parameters
        ----------
        summary: pd.DataFrame
            Chain Ladder summary compilation.

        reserve_dist: pd.DataFrame
            The predicitive distribution of reserve estimates generated via
            bootstrapping. ``reserve_dist`` is a five column DataFrame
            consisting of the simulation number, origin period, the latest
            loss amount for the associated origin period, and the predictive
            distribution of ultimates and reserves.

        sims_data: pd.DataFrame
            A DataFrame consiting of all simulated values an intermediate
            fields. When a large number of bootstrap iterations are run,
            ``sims_data`` will be correspondingly large. The fields include:
            **dev**: The simulated development period.
            **incr**: The actual incremental loss amount obtain from the fitted
            triangle.
            **incr_sqrt**: The square root of incr.
            **l_act_cum**: The latest actual cumulative loss amount for
            dev/origin.
            **l_act_dev**: The latest dev period with actual losses for a
            given origin period.
            **ldf**: Loss development factors computed on syntehtic triangle
            data.
            **origin**: The simulated origin period.
            **rectype**: Whether the dev/origin combination represent actual
            or forecast data in the squared triangle.
            **resid**: The resampled adjusted residuals if ``parametric=False``,
            otherwise a random sampling from a normal distribution with mean
            zero and variance based on the variance of the adjusted residuals.
            **samp_cum**: A syntehtic cumulative loss amount.
            **samp_incr**: A synthetic incremental loss amount.
            **sim**: The simulation number.
            **var**: The variance, computed as scale_param * samp_incr.
            **sign**: The sign of samp_incr.
            **param2/param1**: Parameters for the process error distribution.
            **final_incr**: Final simulated incremetnal loss amount after
            the incorporation of process error.
            **final_cum**: Final simulated cumulative loss amount after
            the incorporation of process error.

        tri: trikit.triangle.CumTriangle
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

        scale_param: float
            The the sum of the squared unscaled Pearson residuals over the
            triangle's degrees of freedom.

        unscaled_residuals: pd.DataFrame
            The unscaled residuals.

        adjusted_residuals: pd.DataFrame
            The adjusted residuals.

        sampling_dist: np.ndarray
            Same as ``adjusted_residuals`` but as a numpy array with
            NaN's and 0's removed.

        fitted_tri_cum: pd.DataFrame
            The cumulative triangle fit using backwards recursion.

        fitted_tri_incr: pd.DataFrame
            The incremental triangle fit using backwards recursion.

        kwargs: dict
            Additional keyword arguments passed into
            ``BootstrapChainLadder``'s ``run`` method.
        """
        super().__init__(summary=summary, tri=tri, ldfs=ldfs, cldfs=cldfs,
                         latest=latest, maturity=maturity, ultimates=ultimates,
                         reserves=reserves, trisqrd=trisqrd, **kwargs)

        self.unscaled_residuals = unscaled_residuals
        self.adjusted_residuals = adjusted_residuals
        self.fitted_tri_incr = fitted_tri_incr
        self.fitted_tri_cum = fitted_tri_cum
        self.sampling_dist = sampling_dist
        self.reserve_dist = reserve_dist
        self.scale_param = scale_param
        self.sims_data = sims_data
        self.ultimates = ultimates
        self.reserves = reserves
        self.maturity = maturity
        self.summary = summary
        self.trisqrd = trisqrd
        self.latest = latest
        self.cldfs = cldfs
        self.ldfs = ldfs
        self.tail = 1.0
        self.tri = tri

        if kwargs is not None:
            for key_ in kwargs:
                setattr(self, key_, kwargs[key_])

        # Properties.
        self._aggregate_distribution = None
        self._origin_distribution = None
        self._residuals_detail = None
        self._fit_assessment = None

        pctlfields_ = [i for i in self.summary.columns if i.endswith("%")]
        pctlfmts_ = {i:"{:.0f}".format for i in pctlfields_}
        self._summspecs.update(pctlfmts_)


    @property
    def origin_distribution(self):
        """
        Return distribution of bootstrapped ultimates/reserves by
        origin period.

        Returns
        -------
        pd.DataFrame
        """
        if self._origin_distribution is None:
            keepcols_ = ["latest", "ultimate", "reserve"]
            self._origin_distribution = self.reserve_dist.groupby(
                ["sim", "origin"], as_index=False)[keepcols_].sum()
        return(self._origin_distribution)


    @property
    def aggregate_distribution(self):
        """
        Return distribution of boostrapped ultimates/reserves aggregated
        over all origin periods.

        Returns
        -------
        pd.DataFrame
        """
        if self._aggregate_distribution is None:
            self._aggregate_distribution = self.origin_distribution.drop("origin", axis=1)
            self._aggregate_distribution = self._aggregate_distribution.groupby("sim", as_index=False).sum()
        return(self._aggregate_distribution)


    @property
    def fit_assessment(self):
        """
        Return a summary assessing the fit of the parametric model used for
        bootstrap resampling. Applicable when ``parametric`` argument to
        is True. Returns a dictionary with keys ``kstest``, ``anderson``,
        ``shapiro``, ``skewtest``, ``kurtosistest`` and ``normaltest``,
        corresponding to statistical tests available in ``scipy.stats``.

        Returns
        -------
        dict
        """
        if self._fit_assessment is None:
            if not self.parametric:
                mean_ = self.sampling_dist.mean()
                stddev_ = self.sampling_dist.std(ddof=1)
                dist_ = stats.norm(loc=mean_, scale=stddev_)
                D, p_ks = stats.kstest(self.sampling_dist, dist_.cdf)
                W, p_sw = stats.shapiro(self.sampling_dist)
                Z, p_sk = stats.skewtest(self.sampling_dist, axis=0, nan_policy="omit")
                K, p_kt = stats.kurtosistest(self.sampling_dist, axis=0, nan_policy="omit")
                S, p_nt = stats.normaltest(self.sampling_dist, axis=0, nan_policy="omit")
                A, crit, sig = stats.anderson(self.sampling_dist, dist="norm")
                self._fit_assessment = {
                    "kstest":{"statistic":D, "p-value":p_ks},
                    "anderson":{"statistic":A, "critical_values":crit, "significance_levels":sig},
                    "shapiro":{"statistic":W, "p-value":p_sw},
                    "skewtest":{"statistic":Z, "p-value":p_sk},
                    "kurtosistest":{"statistic":K, "p-value":p_kt},
                    "normaltest":{"statistic":S, "p-value":p_nt},
                }
        return(self._fit_assessment)


    @property
    def residuals_detail(self):
        """
        Summary statistics based on triangle residuals.

        Returns
        -------
        pd.DataFrame
        """
        if self._residuals_detail is None:
            if not self.parametric:
                unscaled = self.unscaled_residuals.values.ravel()
                adjusted = self.adjusted_residuals.values.ravel()
                unscaled = unscaled_[~np.isnan(unscaled)]
                adjusted = adjusted_[~np.isnan(adjusted)]
                unscaled = unscaled_[unscaled!=0]
                adjusted = adjusted_[adjusted!=0]
                unscaled_size = unscaled.size
                unscaled_sum = unscaled.sum(axis=0)
                unscaled_ssqr = np.sum(unscaled_**2, axis=0)
                unscaled_min  = unscaled.min(axis=0)
                unscaled_max  = unscaled.max(axis=0)
                unscaled_mean = unscaled.mean(axis=0)
                unscaled_skew = stats.skew(unscaled, axis=0, nan_policy="omit")
                unscaled_mode = stats.mode(unscaled, axis=0, nan_policy="omit").mode[0]
                unscaled_cvar = stats.variation(unscaled, axis=0, nan_policy="omit")
                unscaled_kurt = stats.kurtosis(unscaled, axis=0, nan_policy="omit")
                unscaled_var  = unscaled_.var(ddof=1, axis=0)
                unscaled_stddev = unscaled_.std(ddof=1, axis=0)
                unscaled_med  = np.median(unscaled, axis=0)
                adjusted_size = adjusted_.size
                adjusted_sum  = adjusted_.sum(axis=0)
                adjusted_ssqr = np.sum(adjusted**2, axis=0)
                adjusted_min = adjusted_.min(axis=0)
                adjusted_max  = adjusted_.max(axis=0)
                adjusted_mean = adjusted_.mean(axis=0)
                adjusted_skew = stats.skew(adjusted, axis=0, nan_policy="omit")
                adjusted_mode = stats.mode(adjusted, axis=0, nan_policy="omit").mode[0]
                adjusted_cvar = stats.variation(adjusted, axis=0, nan_policy="omit")
                adjusted_kurt = stats.kurtosis(adjusted, axis=0, nan_policy="omit")
                adjusted_var  = adjusted_.var(ddof=1, axis=0)
                adjusted_stddev = adjusted_.std(ddof=1, axis=0)
                adjusted_med  = np.median(adjusted, axis=0)
                self._residuals_detail = pd.DataFrame({
                    "unscaled":[
                        unscaled_size, unscaled_sum , unscaled_ssqr, unscaled_min,
                        unscaled_max,  unscaled_mean, unscaled_skew, unscaled_mode,
                        unscaled_cvar, unscaled_kurt, unscaled_var , unscaled_stdv,
                        unscaled_med,
                    ],
                    "adjusted":[
                        _adjusted_size, adjusted_sum , adjusted_ssqr, adjusted_min,
                        _adjusted_max,  adjusted_mean, adjusted_skew, adjusted_mode,
                        _adjusted_cvar, adjusted_kurt, adjusted_var , adjusted_stdv,
                        _adjusted_med,
                    ],
                },
                    index=[
                        "size", "sum", "sum_of_squares", "minimum", "maximum", "mean",
                        "skew", "mode", "cov", "kurtosis", "variance",
                        "standard_deviation", "median"
                    ]
                )
        return(self._residuals_detail)


    def _bs_data_transform(self,  q):
        """
        Starts with ``BaseChainLadderResult``'s ``_data_transform``, and
        performs additional pre-processing in order to generate plot of
        bootstrapped reserve ranges by origin period.

        Returns
        -------
        pd.DataFrame
        """
        data = self._data_transform()
        dfsims = self._get_quantile(q=q, two_sided=True)
        data = pd.merge(data, dfsims, how="outer", on=["origin", "dev"])
        pctl_hdrs = [i for i in dfsims.columns if i not in ("origin", "dev")]
        for hdr_ in pctl_hdrs:
            data[hdr_] = np.where(
                data["rectype"].values=="actual", np.NaN, data[hdr_].values
            )

        # Determine the first forecast period by origin, and set q-fields to actuals.
        data["_ff"] = np.where(
            data["rectype"].values=="forecast", data["dev"].values, data["dev"].values.max() + 1)
        data["_minf"] = data.groupby(["origin"])["_ff"].transform("min")
        for hdr_ in pctl_hdrs:
            data[hdr_] = np.where(
                np.logical_and(data["rectype"].values=="forecast", data["_minf"].values==data["dev"].values),
                data["value"].values, data[hdr_].values
            )

        data = data.drop(["_ff", "_minf"], axis=1)
        dfv = data[["origin", "dev", "rectype", "value"]]
        dfl = data[["origin", "dev", "rectype", pctl_hdrs[0]]]
        dfu = data[["origin", "dev", "rectype", pctl_hdrs[-1]]]
        dfl["rectype"] = pctl_hdrs[0]
        dfl = dfl.rename({pctl_hdrs[0]:"value"}, axis=1)
        dfu["rectype"] = pctl_hdrs[-1]
        dfu = dfu.rename({pctl_hdrs[-1]:"value"}, axis=1)
        return(pd.concat([dfv, dfl, dfu]).sort_index().reset_index(drop=True))


    def plot(self, q=.90, actuals_color="#334488", forecasts_color="#FFFFFF",
             fill_color="#FCFCB1", fill_alpha=.75, axes_style="darkgrid",
             context="notebook", col_wrap=4, hue_kws=None, **kwargs):
        """
        Generate exhibit representing the distribution of reserve estimates
        resulting from bootstrap resampling, along with percentiles from the
        distribution given by ``q``, the percentile(s) of interest.

        Parameters
        ----------
        q: float in range of [0,1]
            two_sided percentile interval to highlight, which must be between
            0 and 1 inclusive. For example, when ``q=.90``, the 5th and
            95th percentile of the ultimate/reserve distribution will be
            highlighted in the exhibit $(\frac{1 - q}{2}, \frac(1 + q}{2})$.

        actuals_color: str
            A color name or hexidecimal code used to represent actual
            observations. Defaults to "#00264C".

        forecasts_color: str
            A color name or hexidecimal code used to represent forecast
            observations. Defaults to "#FFFFFF".

        fill_color: str
            A color name or hexidecimal code used to represent the fill color
            between percentiles of the ultimate/reserve bootstrap
            distribution as specified by ``q``. Defaults to "#FCFCB1".

        fill_alpha: float
            Control transparency of ``fill_color`` between upper and lower
            percentile bounds of the ultimate/reserve distribution. Defaults
            to .75.

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

        hue_kws: dictionary of param:list of values mapping
            Other keyword arguments to insert into the plotting call to let
            other plot attributes vary across levels of the hue variable
            (e.g. the markers in a scatterplot). Each list of values should
            have length 4, with each index representing aesthetic
            overrides for forecasts, actuals, lower percentile and upper
            percentile renderings respectively. Defaults to ``None``.

        kwargs: dict
            Additional styling options for scatter points. This should include
            additional options accepted by ``plt.plot``.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_context(context)

        data = self._bs_data_transform(q=q)

        pctl_hdrs = sorted(
            [i for i in data["rectype"].unique() if i not in ("actual", "forecast")]
        )

        with sns.axes_style(axes_style):
            huekwargs = dict(
                marker=["o", "o", None, None,], markersize=[6, 6, None, None,],
                color=["#000000", "#000000", "#000000", "#000000",],
                fillstyle=["full", "full", "none", "none",],
                markerfacecolor=[forecasts_color, actuals_color, None, None,],
                markeredgecolor=["#000000", "#000000", None, None,],
                markeredgewidth=[.50, .50, None, None,],
                linestyle=["-", "-", "-.", "--",], linewidth=[.475, .475, .625, .625,],
            )

            if hue_kws is not None:
                # Determine whether the length of each element of hue_kws is 4.
                if all(len(hue_kws[i])==4 for i in hue_kws):
                    huekwargs.update(hue_kws)
                else:
                    warnings.warn("hue_kws overrides not correct length - Ignoring.")

            titlestr = "bootstrap chainladder ultimate range projections"

            grid = sns.FacetGrid(
                data, col="origin", hue="rectype", hue_kws=huekwargs,
                col_wrap=col_wrap, margin_titles=False, despine=True,
                sharex=False, sharey=False,
                hue_order=["forecast", "actual", pctl_hdrs[0], pctl_hdrs[-1]]
            )

            mean_ = grid.map(plt.plot, "dev", "value",)
            grid.set_axis_labels("", "")
            grid.set(xticks=data["dev"].unique().tolist())
            grid.set_titles("", size=5)
            # grid_.set_titles("{col_name}", size=9)
            grid.set_xticklabels(data["dev"].unique().tolist(), size=8)

            # Change ticklabel font size and place legend on each facet.
            for ii, _ in enumerate(grid.axes):
                ax_ = grid.axes[ii]
                origin_ = str(self.tri.origins.get(ii))
                legend_ = ax_.legend(
                    loc="upper left", fontsize="x-small", frameon=True,
                    fancybox=True, shadow=False, edgecolor="#909090",
                    framealpha=1, markerfirst=True,)
                legend_.get_frame().set_facecolor("#FFFFFF")

                # Include thousandths separator on each facet's y-axis label.
                ax_.set_yticklabels(
                    ["{:,.0f}".format(i) for i in ax_.get_yticks()], size=8
                )

                ax_.annotate(
                    origin_, xy=(.85, .925), xytext=(.85, .925), xycoords='axes fraction',
                    textcoords='axes fraction', fontsize=9, rotation=0, color="#000000",
                )

                # Fill between upper and lower range bounds.
                axc = ax_.get_children()
                lines_ = [jj for jj in axc if isinstance(jj, matplotlib.lines.Line2D)]
                xx = [jj._x for jj in lines_ if len(jj._x)>0]
                yy = [jj._y for jj in lines_ if len(jj._y)>0]
                x_, lb_, ub_ = xx[0], yy[-2], yy[-1]
                ax_.fill_between(x_, lb_, ub_, color=fill_color, alpha=fill_alpha)

                # Draw border around each facet.
                for _, spine_ in ax_.spines.items():
                    spine_.set(visible=True, color="#000000", linewidth=.50)

        plt.show()


    def hist(self, color="#FFFFFF", axes_style="darkgrid", context="notebook",
             col_wrap=4, **kwargs):
        """
        Generate visual representation of full predicitive distribtion
        of ultimates/reserves by origin and in aggregate. Additional
        options to style seaborn's distplot can be passed as keyword
        arguments.

        Parameters
        ----------
        color: str
            Determines histogram color in each facet. Can also be specified as
            a key-value pair in ``kwargs``.
        axes_style: str
            Aesthetic style of plots. Defaults to "darkgrid". Other options
            include {whitegrid, dark, white, ticks}.

        context: str
            Set the plotting context parameters. According to the seaborn
            documentation, This affects things like the size of the labels,
            lines, and other elements of the plot, but not the overall style.
            Defaults to ``"notebook"``. Additional options include
            {paper, talk, poster}.

        col_wrap: int
            The maximum number of origin period axes to have on a single row
            of the resulting FacetGrid. Defaults to 5.

        kwargs: dict
            Dictionary of optional matplotlib styling parameters.

        """
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_context(context)

        which_ = "ultimate"
        data0 = self.sims_data[["sim", "origin", "dev", "rectype", "latest", "ultimate", "reserve",]]
        data0 = data0[(data0["dev"]==data0["dev"].max()) & (data0["rectype"]=="forecast")].reset_index(drop=True)
        tot_origin = data0["origin"].max() + 1
        data0 = data0.drop(["dev", "rectype", "latest"], axis=1)

        # Include additional origin representing aggregate distribution.
        data1 = data0.groupby("sim", as_index=False)[["ultimate", "reserve"]].sum()
        data1["origin"] = tot_origin
        data = pd.concat([data0, data1])

        # Get mean, min and max ultimate and reserve by origin.
        med_data = data.groupby("origin", as_index=False)[["ultimate", "reserve"]].median().rename(
            {"ultimate":"med_ult", "reserve":"med_res"}, axis=1).set_index("origin")
        min_data = data.groupby("origin", as_index=False)[["ultimate", "reserve"]].min().rename(
            {"ultimate":"min_ult", "reserve":"min_res"}, axis=1).set_index("origin")
        max_data = data.groupby("origin", as_index=False)[["ultimate", "reserve"]].max().rename(
            {"ultimate":"max_ult", "reserve":"max_res"}, axis=1).set_index("origin")
        dfmetrics = functools.reduce(lambda df1, df2: df1.join(df2), (med_data, min_data, max_data))
        dfmetrics = dfmetrics.reset_index(drop=False).applymap(lambda v: 0 if v<0 else v)


        with sns.axes_style(axes_style):

            pltkwargs = {
                "color":color, "bins":20, "edgecolor":"#484848", "alpha":1.,
                "linewidth":.45,
            }

            if kwargs is not None:
                pltkwargs.update(kwargs)

            grid = sns.FacetGrid(
                data, col="origin", col_wrap=col_wrap, margin_titles=False,
                despine=True, sharex=False, sharey=False,
            )
            hists_ = grid.map(
                plt.hist, "ultimate", **pltkwargs
            )

            grid.set_axis_labels("", "")
            grid.set_titles("", size=6)

            # Change ticklabel font size and place legend on each facet.
            uniq_origins = np.sort(data.origin.unique())
            med_hdr = "med_ult" if which_.startswith("ult") else "med_res"
            min_hdr = "min_ult" if which_.startswith("ult") else "min_res"
            max_hdr = "max_ult" if which_.startswith("ult") else "max_res"

            for ii, ax_ in enumerate(grid.axes.flatten()):

                origin_ = uniq_origins[ii]
                xmin = np.max([0, dfmetrics[dfmetrics.origin==origin_][min_hdr].item()])
                xmax = dfmetrics[dfmetrics.origin==origin_][max_hdr].item() * 1.025
                xmed = dfmetrics[dfmetrics.origin==origin_][med_hdr].item()
                origin_str = "total {}".format(which_) if origin_==tot_origin else "{} {}".format(origin_, which_)
                ax_.set_xlim([0, xmax])
                ax_.axvline(xmed)
                ax_.grid(True)

                ymedloc = max(rect.get_height() for rect in ax_.patches) * .30

                ax_.tick_params(
                    axis="x", which="both", bottom=True, top=False, labelbottom=True
                )
                ax_.set_xticklabels(
                    ["{:,.0f}".format(jj) for jj in ax_.get_xticks()], size=7
                )
                ax_.annotate(
                    origin_str, xy=(.85, .925), xytext=(.65, .925), xycoords='axes fraction',
                    textcoords='axes fraction', fontsize=8, rotation=0, color="#000000",
                )
                ax_.annotate(
                    "median = {:,.0f}".format(xmed), (xmed, ymedloc), xytext=(7.5, 0),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6,
                    rotation=90, color="#4b0082"
                )

                # Draw border around each facet.
                for _, spine_ in ax_.spines.items():
                    spine_.set(visible=True, color="#000000", linewidth=.50)

        plt.show()


    def _get_quantile(self, q, two_sided=True, interpolation="linear"):
        """
        Return percentile of bootstrapped ultimate or reserve range
        distribution as specified by ``q``.

        Parameters
        ----------
        q: float in range of [0,1] (or sequence of floats)
            Percentile to compute, which must be between 0 and 1 inclusive.

        two_sided: bool
            Whether the two_sided interval should be returned. For example, if
            ``two_sided==True`` and ``q=.95``, then the 2.5th and 97.5th
            quantiles of the predictive reserve distribution will be returned
            [(1 - .95) / 2, (1 + .95) / 2]. When False, only the specified
            quantile(s) will be computed.

        interpolation: {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use when
            the desired quantile lies between two data points i < j:

                - linear: i + (j - i) * fraction, where fraction is the fractional
                part of the index surrounded by i and j.
                - lower: i.
                - higher: j.
                - nearest: i or j, whichever is nearest.
                - midpoint: (i + j) / 2.

        Returns
        -------
        pd.DataFrame
        """
        dfsims = self.sims_data[["origin", "dev", "ultimate"]]
        pctl = np.asarray([q] if isinstance(q, (float, int)) else q)
        if np.any(np.logical_and(pctl <= 1, pctl >= 0)):
            if two_sided:
                pctlarr = np.sort(np.unique(np.append((1 - pctl) / 2, (1 + pctl) / 2)))
            else:
                pctlarr = np.sort(np.unique(pctl))
        else:
            raise ValueError("Values for percentiles must fall between [0, 1].")

        # Initialize DataFrame for percentile data.
        pctlfmt = ["{:.5f}".format(i).rstrip("0").rstrip(".") + "%" for i in 100 * pctlarr]
        dfpctl = dfsims.groupby(["origin", "dev"]).aggregate(
            "quantile", q=.50, interpolation=interpolation)
        dfpctl = dfpctl.rename({"ultimate":"50%"}, axis=1)
        dfpctl.columns.name = None

        for q_, pctlstr_ in zip(pctlarr, pctlfmt):
            if q_!=.50:
                df_ = dfsims.groupby(["origin", "dev"]).aggregate(
                    "quantile", q=q_, interpolation=interpolation)
                df_ = df_.rename({"ultimate":pctlstr_}, axis=1)
                df_.columns.name = None
                dfpctl = dfpctl.join(df_)

        if .50 not in pctl:
            dfpctl = dfpctl.drop("50%", axis=1)

        return(dfpctl.reset_index(drop=False).sort_index())



    def __str__(self):
        return(self.summary.to_string(formatters=self._summspecs))


    # def __repr__(self):
    #     # pctls_ = [i for i in self.summary.columns if i.endswith("%")]
    #     # pctlfmts_ = {i:"{:.0f}".format for i in pctls_}
    #     # formats_ = {"ultimate":"{:.0f}".format, "reserve":"{:.0f}".format,
    #     #             "latest":"{:.0f}".format, "cldf":"{:.5f}".format,}
    #     return(self.summary.to_string(formatters=self._summspecs))
