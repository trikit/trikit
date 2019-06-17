"""
_MackChainLadder class definition.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from ..chainladder import _BaseChainLadder, _BaseChainLadderResult



class _MackChainLadder(_BaseChainLadder):
    """
    Perform Mack Chain Ladder method. The predicition variance is comprised
    of the estimation variance and the process variance. Estimation
    variance arises from the inability to accurately define the distribution
    from which past events have been generated. Process variance arises from
    the inability to accurately predict which single outcome from the
    distribution will occur at a given time. The predicition error is
    specified as the standard deviation of the forecast.


    References
    ----------
    - Mack, T. (1993), *Distribution Free Calculation of the Standard Error of
    Chain Ladder Reserve Estimates*, ASTIN Bulletin 23, 213-225.
    - Mack, T. (1999), *The Standard Error of Chain Ladder Reserve Estimates:
    Recursive Calculation and Inclusion of a Tail Factor*, ASTIN Bulletin
    29(2), 361-366.
    - Murphy, D. (2007), *Chain Ladder Reserve Risk Estimators*, CAS E-Forum.
    """
    def __init__(self, cumtri):
        """
        Parameters
        ----------
        cumtri: triangle._CumTriangle
            A cumulative.CumTriangle instance
        """

        super().__init__(cumtri)

        # properties
        self._parameter_error = None
        self._process_error = None
        self._inverse_sums = None
        self._originref = None
        self._devpref = None
        #self._devpvar = None
        self._mseptot = None
        self._rmsepi = None
        self._msepi = None



    def __call__(self, alpha=1, tail=1.0):
        """
        Compute the reserve risk associated with Chain Ladder estimates.
        """
        ldfs_ = self._ldfs(alpha=alpha, tail=tail)
        cldfs_ = self._cldfs(ldfs=ldfs_)
        ultimates_ = self._ultimates(cldfs=cldfs_)
        reserves_ = self._reserves(ultimates=ultimates_)
        maturity_ = self.tri.maturity.astype(np.str)
        latest_ = self.tri.latest_by_origin






        summcols = ["maturity", "latest", "cldf", "ultimate", "reserve"]
        summDF   = pd.DataFrame(columns=summcols, index=self.tri.index)
        summDF["maturity"]  = self.tri.maturity.astype(np.str)
        summDF["latest"]    = self.tri.latest_by_origin
        summDF["cldf"]      = cldfs.values[::-1]
        summDF["ultimate"]  = ults
        summDF["reserve"]   = ibnr
        self._summary['RMSEP']        = self.rmsepi
        self._summary['CV']           = self.rmsepi/self.reserves
        self._summary["NORM_95%_LB"]  = self.reserves - (1.96 * self.rmsepi)
        self._summary["NORM_95%_UB"]  = self.reserves + (1.96 * self.rmsepi)
        self._summary["LNORM_95%_LB"] = np.exp(mu_i - 1.96 * sigma_i)
        self._summary["LNORM_95%_UB"] = np.exp(mu_i + 1.96 * sigma_i)
        self._summary.loc['TOTAL']    = self._summary.sum()
        summDF.loc["total"] = summDF.sum()

        # Set to NaN columns that shouldn't be summed.
        summDF.loc["total", "maturity"] = ""
        summDF.loc["total", "cldf"]     = np.NaN
        summDF = summDF.reset_index().rename({"index":"origin"}, axis="columns")
        return(None)





    def _ldfs(self, alpha=1, tail=1.0):
        """
        Compute Mack LDFs based on provided ``alpha`` and ``tail``.

        Parameters
        ----------
        alpha: {0, 1, 2}
            The parameter specifying the approach used to compute the LDF
            patterns used in the Mack Chain Ladder. ``alpha=1`` gives the
            historical chain ladder age-to-age factors, ``alpha=0`` gives the
            straight average of the observed individual development factors
            and ``alpha=2`` is the result of an ordinary
            regression of $C_{i, k + 1}$ against $C_{i, k}$ with intercept
            0.

        tail: float
            Tail factor. Defaults to 1.0.

        Returns
        -------
        pd.Series
        """
        if alpha not in (0, 1 ,2,):
            raise ValueError("Invalid `alpha` specification: {}".format(alpha))
        else:
            common_ = (self.tri.pow(alpha) * self.tri.a2aind)
            common_ = common_.dropna(axis=0, how="all").dropna(axis=1, how="all")
            ldfs_ = (common_ * self.tri.a2a).sum() / common_.sum()
            tindx_ = ldfs_.index.max() + 1
            ldfs_ = ldfs_.append(pd.Series(data=[tail], index=[tindx_]))
        return(pd.Series(data=ldfs_, index=ldfs_.index, name="ldf"))


    def _devpvar(self, alpha=1, tail=1.0):
        """
        Return the development period variance estimator, the sum of the
        squared deviations of losses at the end of the development period from
        the Chain ladder predictions given the losses at the beginning of the
        period, all divided by n - 1, where n is the number of terms in the
        summation.
        """
        ldfs_ = self._ldfs(alpha=alpha, tail=tail)
        diffs_ = self.tri.a2a.subtract(ldfs_, axis=1).pow(2).replace(0, np.NaN)
        prod_ = (diffs_ * self.tri.pow(alpha))
        prod_ = prod_.dropna(axis=0, how="all").dropna(axis=1, how="all").sum()
        ratio_ = pd.Series(data=(1 / (self.tri.shape[0] - prod_.index - 1)), index=prod_.index)
        devpvar_ = ratio_ * prod_

        # Add  `n-1` development period variance term.
        tindx_ = devpvar_.index.max() + 1
        final_ = np.min([
            devpvar_.values[-1]**2 / devpvar_.values[-2],
            np.min([devpvar_.values[-2], devpvar_.values[-1]])])
        devpvar_ = devpvar_.append(pd.Series(data=[final_], index=[tindx_]))
        return(pd.Series(data=devpvar_, index=devpvar_.index, name="devpvar"))







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




    @property
    def process_error(self):
        """
        Process error (forecast error) calculation. The process error
        component originates from the stochastic movement of the process.
        Returns a pandas Series containing estimates of process variance by
        origin year.
        """
        if self._process_error is None:
            lastcol, pelist = self.tri.columns.size-1, list()
            for rindx in self.tri.rlvi.index:
                iult = self.ultimates[rindx]
                ilvi = self.tri.rlvi.loc[rindx,:].col_offset
                ilvc = self.tri.rlvi.loc[rindx,:].dev
                ipe  = 0
                if ilvi<lastcol:
                    for dev in self.trisqrd.loc[rindx][ilvi:lastcol].index:
                        ipe+=(self.devpref.loc[dev,'RATIO'] / self.trisqrd.loc[rindx,dev])
                    ipe*=(iult**2)
                pelist.append((rindx,ipe))

            # Convert list of tuples into Series object.
            indx, vals = zip(*pelist)
            self._process_error = \
                pd.Series(data=vals, index=indx, name="process_error")
        return(self._process_error)




    @property
    def parameter_error(self):
        """
        Estimation error (parameter error) reflects the uncertainty in
        the estimation of the parameters.
        """
        if self._parameter_error is None:
            lastcol, pelist = self.tri.columns.size-1, list()
            for i in enumerate(self.tri.index):
                ii, rindx = i[0], i[1]
                iult = self.ultimates[rindx]
                ilvi = self.tri.rlvi.loc[rindx,:].col_offset
                ilvc = self.tri.rlvi.loc[rindx,:].dev
                ipe  = 0
                if ilvi<lastcol:
                    for k in range(ilvi,lastcol):
                        ratio  = self.devpref[self.devpref['indx']==k]['ratio'].values[0]
                        invsum = self.devpref[self.devpref['indx']==k]['inv_sum'].values[0]
                        ipe+=(ratio * invsum)
                    ipe*=iult**2
                else:
                    ipe = 0
                pelist.append((rindx, ipe))
            # Convert list of tuples into Series object.
            indx, vals = zip(*pelist)
            self._parameter_error = \
                pd.Series(data=vals, index=indx, name="parameter_error")
        return(self._parameter_error)




    @property
    def covariance_term(self):
        """
        Used to derive the conditional mean squared error of total
        reserve prediction. MSE_(i,j) is non-zero only for cells
        in which i < j (i.e., the
        :return:
        """
        pass


    @property
    def msepi(self):
        """
        Return the mean squared error of predicition by origin year.
        Does not contain estimate for total MSEP.
        MSE_i = process error + parameter error
        """
        if self._msepi is None:
            self._msepi = self.process_error + self.parameter_error
        return(self._msepi)


    @property
    def rmsepi(self):
        """
        Return the root mean squared error of predicition by origin
        year. Does not contain estimate for total MSEP.

            MSE_i = process error + parameter error
        """
        if self._rmsepi is None:
            self._rmsepi = np.sqrt(self.msepi)
        return(self._rmsepi)




    @property
    def summary(self):
        """
        Return a DataFrame containing summary statistics resulting
        from applying the development method to tri, in addition
        to Mack-generated range estimates.
        """
        if self._summary is None:
            self._summary = \
                pd.DataFrame(
                    columns=[
                        "LATEST","CLDF","EMERGENCE","ULTIMATE","RESERVE","RMSEP",
                        "CV","NORM_95%_LB","NORM_95%_UB","LNORM_95%_LB","LNORM_95%_UB"
                        ], index=self.tri.index
                    )

            # Initialize lognormal confidence interval parameters.
            sigma_i = np.log(1 + self.msepi/self.reserves**2)
            mu_i    = np.log(self.reserves - .50 * sigma_i)

            # Populate self._summary with existing properties if available.
            self._summary['LATEST']       = self.latest_by_origin
            self._summary['CLDF']         = self.cldfs[::-1]
            self._summary['EMERGENCE']    = 1/self.cldfs[::-1]
            self._summary['ULTIMATE']     = self.ultimates
            self._summary['RESERVE']      = self.reserves
            self._summary['RMSEP']        = self.rmsepi
            self._summary['CV']           = self.rmsepi/self.reserves
            self._summary["NORM_95%_LB"]  = self.reserves - (1.96 * self.rmsepi)
            self._summary["NORM_95%_UB"]  = self.reserves + (1.96 * self.rmsepi)
            self._summary["LNORM_95%_LB"] = np.exp(mu_i - 1.96 * sigma_i)
            self._summary["LNORM_95%_UB"] = np.exp(mu_i + 1.96 * sigma_i)
            self._summary.loc['TOTAL']    = self._summary.sum()

            # Set CLDF Total value to `NaN`.
            self._summary.loc["TOTAL","CLDF","EMERGENCE"] = np.NaN

        return(self._summary)








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


class _MackChainLadderResult(_BaseChainLadderResult):
    """
    Curated output resulting from ``_BootstrapChainLadder``'s ``run`` method.
    """
    def __init__(self, summary, reserve_dist, sims_data, tri, ldfs, cldfs,
                 latest, maturity, ultimates, reserves, scale_param,
                 unscaled_residuals, adjusted_residuals, sampling_dist,
                 fitted_tri_cum, fitted_tri_incr, **kwargs):
        """
        Container class for ``BootstrapChainLadder``'s output.

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

        tri: trikit.triangle._CumTriangle
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
            ``_BootstrapChainLadder``'s ``run`` method.
        """
        super().__init__(summary=summary, tri=tri, ldfs=ldfs, cldfs=cldfs,
                         latest=latest, maturity=maturity, ultimates=ultimates,
                         reserves=reserves, **kwargs)

        # self.unscaled_residuals = unscaled_residuals
        # self.adjusted_residuals = adjusted_residuals
        # self.fitted_tri_incr = fitted_tri_incr
        # self.fitted_tri_cum = fitted_tri_cum
        # self.sampling_dist = sampling_dist
        # self.reserve_dist = reserve_dist
        # self.scale_param = scale_param
        # self.sims_data = sims_data
        # self.ultimates = ultimates
        # self.reserves = reserves
        # self.maturity = maturity
        # self.summary = summary
        # self.latest = latest
        # self.cldfs = cldfs
        # self.ldfs = ldfs
        # self.tail = 1.0
        # self.tri = tri

        if kwargs is not None:
            for key_ in kwargs:
                setattr(self, key_, kwargs[key_])

        # Properties.
        # self._residuals_detail = None
        # self._fit_assessment = None
        # self._origindist = None
        # self._aggdist = None

        pctlfields_ = [i for i in self.summary.columns if i.endswith("%")]
        pctlfmts_ = {i:"{:.0f}".format for i in pctlfields_}
        self.summspecs = {"ultimate":"{:.0f}".format, "reserve":"{:.0f}".format,
                          "latest":"{:.0f}".format, "cldf":"{:.5f}".format,}
        self.summspecs.update(pctlfmts_)



    @staticmethod
    def _nbrbins(data):
        """
        Return an estimate for the appropriate number of histogram bins.
        Bin width is determined using the Freedman–Diaconis rule, where
        width = [ 2 * IQR ] / N^(1/3), N=number of observations and IQR
        the interquartile range of the dataset.

        Parameters
        ----------
        data: np.ndarray
            One-dimensional array.

        Returns
        -------
        int
        """
        data = np.asarray(data, dtype=np.float_)
        IQR = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
        N = data.size
        bw = (2 * IQR) / np.power(N, 1/3)
        datrng = data.max() - data.min()
        return(int((datrng / bw) + 1))

    #
    # @property
    # def aggdist(self):
    #     """
    #     Return aggregate distribution of simulated reserve amounts
    #     over all origin years.
    #
    #     Returns
    #     -------
    #     pd.DataFrame
    #     """
    #     if self._aggdist is None:
    #         keepcols_ = ["latest", "ultimate", "reserve"]
    #         self._aggdist = self.reserve_dist.groupby(
    #             ["sim"], as_index=False)[keepcols_].sum()
    #     return(self._aggdist)
    #
    #
    # @property
    # def origindist(self):
    #     """
    #     Return distribution of simulated loss reserves by origin year.
    #
    #     Returns
    #     -------
    #     pd.DataFrame
    #     """
    #     if self._origindist is None:
    #         keepcols_ = ["latest", "ultimate", "reserve"]
    #         self._origindist = self.reserve_dist.groupby(
    #             ["sim", "origin"], as_index=False)[keepcols_].sum()
    #     return(self._origindist)


    # @property
    # def fit_assessment(self):
    #     """
    #     Return a summary assessing the fit of the parametric model used for
    #     bootstrap resampling. Applicable when ``parametric`` argument to
    #     ``run`` is True. Returns a dictionary with keys ``kstest``,
    #     ``anderson``, ``shapiro``, ``skewtest``, ``kurtosistest`` and
    #     ``normaltest``, corresponding to statistical tests available in
    #     scipy.stats.
    #
    #     Returns
    #     -------
    #     dict
    #     """
    #     if self._fit_assessment is None:
    #         if not self.parametric:
    #             mean_ = self.sampling_dist.mean()
    #             stddev_ = self.sampling_dist.std(ddof=1)
    #             dist_ = stats.norm(loc=mean_, scale=stddev_)
    #             D, p_ks = stats.kstest(self.sampling_dist, dist_.cdf)
    #             W, p_sw = stats.shapiro(self.sampling_dist)
    #             Z, p_sk = stats.skewtest(self.sampling_dist, axis=0, nan_policy="omit")
    #             K, p_kt = stats.kurtosistest(self.sampling_dist, axis=0, nan_policy="omit")
    #             S, p_nt = stats.normaltest(self.sampling_dist, axis=0, nan_policy="omit")
    #             A, crit, sig = stats.anderson(self.sampling_dist, dist="norm")
    #             self._fit_assessment = {
    #                 "kstest":{"statistic":D, "p-value":p_ks},
    #                 "anderson":{"statistic":A, "critical_values":crit, "significance_levels":sig},
    #                 "shapiro":{"statistic":W, "p-value":p_sw},
    #                 "skewtest":{"statistic":Z, "p-value":p_sk},
    #                 "kurtosistest":{"statistic":K, "p-value":p_kt},
    #                 "normaltest":{"statistic":S, "p-value":p_nt},
    #                 }
    #     return(self._fit_assessment)





    # def plotdist(self, level="aggregate", tc="#FE0000", path=None, **kwargs):
    #     """
    #     Generate visual representation of full predicitive distribtion
    #     of loss reserves in aggregate or by origin. Additional options
    #     to style the histogram can be passed as keyword arguments.
    #
    #     Parameters
    #     ----------
    #     level: str
    #         Set ``level`` to "origin" for faceted plot with predicitive
    #         distirbution by origin year, or "aggregate" for single plot
    #         of predicitive distribution of reserves in aggregate. Default
    #         value is "aggregate".
    #
    #     path: str
    #         If path is not None, save plot to the specified location.
    #         Otherwise, parameter is ignored. Default value is None.
    #
    #     kwargs: dict
    #         Dictionary of optional matplotlib styling parameters.
    #
    #     """
    #     plt_params = {
    #         "alpha":.995, "color":"#FFFFFF", "align":"mid", "edgecolor":"black",
    #         "histtype":"bar", "linewidth":1.1, "orientation":"vertical",
    #         }
    #
    #     if level.lower().strip().startswith(("agg", "tot")):
    #         # bins computed using self._nbrbins if not passed as optional
    #         # keyword argument.
    #         dat = self.aggdist["reserve"].values
    #         plt_params["bins"] = self._nbrbins(data=dat)
    #
    #         # Update plt_params with any optional keyword arguments.
    #         plt_params.update(kwargs)
    #
    #         # Setup.
    #         fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True)
    #         ax.set_facecolor("#1f77b4")
    #         ax.set_title(
    #             "Distribution of Bootstrap Reserve Estimates (Aggregate)",
    #             loc="left", color=tc)
    #         ax.get_xaxis().set_major_formatter(
    #             mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    #         ax.set_xlabel("Reserves"); ax.set_ylabel("Frequency")
    #         ax.hist(dat, **plt_params)
    #
    #     elif level.lower().strip().startswith(("orig", "year")):
    #         dat = self.origindist[["origin", "reserve"]]
    #         sns.set(rc={'axes.facecolor':"#1f77b4"})
    #         g = sns.FacetGrid(dat, col="origin", col_wrap=4, margin_titles=False)
    #         g.map(plt.hist, "reserve", **plt_params)
    #         g.set_titles("{col_name}", color=tc)
    #         g.fig.suptitle("Reserve Distribution by Origin Year", color=tc, weight="bold")
    #         plt.subplots_adjust(top=0.92)
    #     plt.show()
    #
    #
    # def __str__(self):
    #     return(self.summary.to_string(formatters=self.summspecs))
    #
    #
    # def __repr__(self):
    #     # pctls_ = [i for i in self.summary.columns if i.endswith("%")]
    #     # pctlfmts_ = {i:"{:.0f}".format for i in pctls_}
    #     # formats_ = {"ultimate":"{:.0f}".format, "reserve":"{:.0f}".format,
    #     #             "latest":"{:.0f}".format, "cldf":"{:.5f}".format,}
    #     return(self.summary.to_string(formatters=self.summspecs))

