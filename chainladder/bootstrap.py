"""
_BootstrapChainLadder Implementation.

===========================================================
FUTURE ENHANCEMENTS                                       |
===========================================================
- Allow for residuals other than Pearson (Anscombe, Deviance, etc.)
- Enable other distributions for process variance.
- Add staticmethod for neg_handler==2 (needed in bs_samples).

  From scipy nbinom parameterization:

            Mean     = n * p / (1 - p)
            Variance = n * p / (1 - p)^2
            Variance = Mean / (1 - p)

"""
import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from ..chainladder import _BaseChainLadder
from ..triangle import _IncrTriangle, _CumTriangle
from ..utils import totri, _cumtoincr, _incrtocum, _tritotbl






class _BootstrapChainLadder(_BaseChainLadder):
    """
    The purpose of the bootstrap technique is to estimate the predicition
    error of the total reserve estimate and to approximate the predictive
    distribution. It is often impractical to obtain the prediction error
    using an analytical approach due to the complexity of reserve estimators.

    Predicition error is comprised of two components: process error
    and estimation error (Prediction Error = Estimation Error + Process Error).
    The estimation error (parameter error) represents the uncertainty in the
    parameter estimates given that the model is correctly specified. The
    process error is analogous to the variance of a random variable,
    representing the uncertainty in future outcomes.

    The procedure used to generate the predicitive distribution of reserve
    estimates is based on Leong et al. Appendix A, assuming the starting point
    is a triangle of cumulative losses:

    1.  Calculate the all-year volume-weighted age-to-age factors.
    2.  Estimate the fitted historical cumulative paid loss and ALAE
        using the latest diagonal of the original triangle and the
        age-to-age factors from [1] to un-develop the losses.
    3.  Calculate the unscaled Pearson residuals, degrees of freedom
        and scale parameter.
    4.  Calculate the adjusted Pearson residuals.
    5.  Sample with replacement from the adjusted Pearson residuals.
    6.  Calculate the triangle of sampled incremental losses
        (I^ = m + r_adj * sqrt(m)), where I^ = Resampled incremental loss,
        m = Incremental fitted loss (from [2]) and r_adj = Adjusted Pearson
        residuals.
    7.  Using the triangle from [6], project future losses using the
        Chain Ladder method.
    8.  Include Process variance by simulating each incremental future
        loss from a Gamma distribution with mean = I^ and
        variance = I^ * scale parameter.
    9.  Estimate unpaid losses using the Chain Ladder technique.
    10. Repeat for the number of cycles specified.


    The collection of projected ultimates for each origin year over all
    bootstrap cycles comprises the predictive distribtuion of reserve
    estimates.

    Note that the estimate of the distribution of losses assumes
    development is complete by the final development period. This is
    to avoid the complication in modeling a tail factor.
    """
    def __init__(self, cumtri, sel="all-weighted", tail=1.0, neg_handler=1):
        """
        The _BootstrapChainLadder class definition.

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

        neg_handler: int
            If ``neg_handler=1``, then any first development period negative
            cells will be coerced to +1. If ``neg_handler=2``, the minimum
            value in all triangle cells is identified (identified as 'MIN_CELL').
            If MIN_CELL is less than or equal to 0, the equation
            (MIN_CELL + X = +1.0) is solved for X. X is then added to every
            other cell in the triangle, resulting in all triangle cells having
            a value strictly greater than 0.
        """
        super().__init__(cumtri=cumtri, sel=sel, tail=tail)

        self.neg_handler = neg_handler

        # Properties
        self._residuals_detail = None
        self._fit_assessment = None
        self._sampling_dist = None
        self._tri_fit_incr = None
        self._scale_param = None
        self._tri_fit_cum = None
        self._nbr_cells = None
        self._resid_adj = None
        self._resid_us = None
        self._dof = None



    def __call__(self, sims=1000, procdist="gamma", parametric=False,
                 percentiles=[.75, .95], interpolation="linear",
                 random_state=None):
        """
        ``_BootstrapChainLadder`` simulation initializer. Generates predictive
        distribution of ultimate reserve outcomes.

        Parameters
        ----------
        sims: int
            The number of bootstrap simulations to perfrom. Defaults to 1000.

        procdist: str
            The distribution used to incorporate process variance. Currently,
            this can only be set to "gamma". Future state will also allow
            over-dispersed poisson ("odp").

        percentiles: list
            The percentiles to include along with the Chain Ladder point
            estimates when ``summary``=True. Defaults to [.75, .95].

        parametric: bool
            If True, fit standardized residuals to a normal distribution, and
            sample from this parameterized distribution. Otherwise, bootstrap
            procedure samples with replacement from the collection of
            standardized residuals. Defaults to False.

        interpolation: {"linear", "lower", "higher", "midpoint", "nearest"}
            Specifies the interpolation method to use when the desired
            percentile lies between two data points. Defaults to "linear".
            Argument only valid when ``returnas``="summary".

        random_state: np.random.RandomState
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        _BootstrapResult
            Content varies based on ``returnas`` parameter.
        """
        kwdargs    = locals()
        _samples   = self._bs_samples(sims=sims, parametric=parametric, random_state=random_state)
        _ldfs      = self._bs_ldfs(samples_df=_samples)
        _rlvi      = self.tri.rlvi.reset_index().rename({"index":"origin", "dev":"l_act_dev"},axis=1).drop(labels="col_offset", axis=1)
        _combined  = _samples.merge(_ldfs, how="left", on=["sim", "dev"])
        _combined  = _combined.merge(_rlvi, how="left", on=["origin"]).reset_index(drop=True).sort_values(by=["sim", "origin", "dev"])
        _forecasts = self._bs_forecasts(combined_df=_combined)
        _procerror = self._bs_process_error(forecasts_df=_forecasts, procdist=procdist, random_state=random_state)
        _reserves  = self._bs_reserves(process_error_df=_procerror)

        # Compile summary DataFrame.
        pctlarr1 = np.unique(np.array(percentiles))
        if np.all(pctlarr1 <= 1):
            pctlarr1 = 100 * pctlarr1
        pctlarr2 = 100 - pctlarr1
        pctlarr  = np.unique(np.append(pctlarr1, pctlarr2))
        pctlarr.sort()
        pctllist = [i if i < 10 else int(i) for i in pctlarr]
        pctlstrs = [str(i)  + "%" for i in pctllist]
        summcols = ["maturity", "cldf", "latest", "ultimate", "reserve"]
        summdf   = pd.DataFrame(columns=summcols, index=self.tri.index)
        summdf["maturity"] = self.tri.maturity.astype(np.str)
        summdf["cldf"]     = self.cldfs.values[::-1].astype(np.float_)
        summdf["latest"]   = self.tri.latest_by_origin.astype(np.float_)
        summdf["ultimate"] = self.ultimates.astype(np.float_)
        summdf["reserve"]  = self.reserves.astype(np.float_)
        summdf             = summdf.rename({"index":"origin"}, axis=1)
        for pctl, pctlstr in zip(pctllist, pctlstrs):
            summdf[pctlstr] = summdf.index.map(
                lambda v: np.percentile(
                    _reserves["reserve"][_reserves["origin"]==v].values, pctl, interpolation=interpolation
                    )
                )

        # Set to NaN columns that shouldn't be aggregated.
        summdf.loc["total"] = summdf.sum()
        summdf.loc["total", "maturity"] = ""
        summdf.loc["total", "cldf"]     = np.NaN

        # Initialize _BootstrapResult instance.
        kwdargs.pop("self"); kwdargs.pop("random_state")
        result = _BootstrapResult(
            summary_df=summdf, reserves_df=_reserves,
            process_error_df=_procerror, **kwdargs
            )

        # Testing ============================================================]
        # sims          = 100
        # procdist      = "gamma"
        # parametric    = False
        # neg_handler   = 1
        # percentiles   = [.75, .95]
        # interpolation = "linear"
        # returnas      = "summary"
        # random_state  = RandomState(20180516)
        #
        # _samples   = bcl._bs_samples(sims=sims, parametric=parametric, random_state=random_state)
        # _ldfs      = bcl._bs_ldfs(samples=_samples)
        # _rlvi      = bcl.tri.rlvi.reset_index().rename({"index":"origin", "dev":"l_act_dev"},axis=1).drop(labels="col_offset", axis=1)
        # _combined  = _samples.merge(_ldfs, how="left", on=["sim", "dev"])
        # _combined  = _combined.merge(_rlvi, how="left", on=["origin"]).reset_index(drop=True).sort_values(by=["sim", "origin", "dev"])
        # _forecasts = bcl._bs_forecasts(combined_df=_combined)
        # _procerror = bcl._bs_process_error(forecasts_df=_forecasts, procdist=procdist, random_state=random_state)
        # _reserves  = bcl._bs_reserves(process_error_df=_procerror)
        #
        # pctlarr1 = np.unique(np.array(percentiles))
        # if np.all(pctlarr1 <= 1): pctlarr1 = 100 * pctlarr1
        #
        # pctlarr2 = 100 - pctlarr1
        # pctlarr  = np.unique(np.append(pctlarr1, pctlarr2))
        # pctlarr.sort()
        # pctllist = [i if i < 10 else int(i) for i in pctlarr]
        # pctlstrs = [str(i)  + "%" for i in pctllist]
        # summcols = ["maturity", "latest", "cldf", "ultimate", "reserve"]
        #
        # summdf = pd.DataFrame(columns=summcols, index=bcl.tri.index)
        # summdf["maturity"]  = bcl.tri.maturity.astype(np.str)
        # summdf["latest"]    = bcl.tri.latest_by_origin
        # summdf["cldf"]      = bcl.cldfs.values[::-1]
        # summdf["ultimate"]  = bcl.ultimates
        # summdf["reserve"]   = bcl.reserves
        # # End testing ========================================================]
        #
        # summdf = summdf.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        #
        # for pctl, pctlstr in zip(pctllist, pctlstrs):
        #     summdf[pctlstr] = summdf.index.map(
        #         lambda v: np.percentile(
        #             _reserves["reserve"][_reserves["origin"]==v].values, pctl, interpolation=interpolation
        #             )
        #         )
        #
        # result = summdf

        return(result)







    @property
    def nbr_cells(self):
        """
        Compute and return the number of non-NaN cells in self.tri.

        Returns
        -------
        int
            An integer representing the number of non-NaN cells in self.tri.

        """
        if self._nbr_cells is None:
            self._nbr_cells = self.tri.count().sum()
        return(self._nbr_cells)



    @property
    def dof(self):
        """
        Compute and return the target triangle's degress of freedom.

        Returns
        -------
        int
            The degrees of freedom associated with self.tri.
        """
        if self._dof is None:
            self._dof = \
                self.nbr_cells - (self.tri.columns.size-1) + self.tri.index.size
        return(self._dof)



    @property
    def scale_param(self):
        """
        Compute and return the scale parameter, which is the sum of the
        squared unscaled Pearson residuals over the degrees of freedom.

        Returns
        -------
        float
            Scale parameter.
        """
        if self._scale_param is None:
            self._scale_param = (self.resid_us**2).sum().sum() / self.dof
        return(self._scale_param)




    @property
    def tri_fit_cum(self):
        """
        Compute and return the cumulative fitted triangle using backwards
        recursion, starting with the observed cumulative paid-to-date along
        the latest diagonal.

        Returns
        -------
        pd.DataFrame
            The fitted cumulative loss triangle.
        """
        if self._tri_fit_cum is None:
            self._tri_fit_cum = self.tri.copy(deep=True)
            for i in range(self._tri_fit_cum.shape[0]):
                iterrow  = self._tri_fit_cum.iloc[i, :] # same as tri.iloc[i]
                if iterrow.isnull().any():
                    # Find first NaN element in iterrow.
                    nan_hdr  = iterrow.isnull()[iterrow.isnull()==True].index[0]
                    nan_idx  = self._tri_fit_cum.columns.tolist().index(nan_hdr)
                    init_idx = nan_idx - 1
                else:
                    # If here, iterrow is the most mature exposure period.
                    init_idx = self._tri_fit_cum.shape[1] - 1
                # Set to NaN any development periods earlier than init_idx.
                self._tri_fit_cum.iloc[i, :init_idx] = np.NaN
                # Iterate over rows, undeveloping triangle from latest diagonal.
                for j in range(self._tri_fit_cum.iloc[i, :init_idx].size, 0, -1):
                    prev_col_idx = j
                    curr_col_idx = j - 1
                    curr_ldf_idx = j - 1
                    prev_col_val = self._tri_fit_cum.iloc[i, prev_col_idx]
                    curr_ldf_val = self.ldfs[curr_ldf_idx]
                    self._tri_fit_cum.iloc[i, curr_col_idx] = (prev_col_val / curr_ldf_val)
        return(self._tri_fit_cum)



    @property
    def tri_fit_incr(self):
        """
        Compute and return the fitted incremental triangle.

        Returns
        -------
        pd.DataFrame
            The fitted incremental loss triangle.

        """
        if self._tri_fit_incr is None:
            self._tri_fit_incr = _cumtoincr(self.tri_fit_cum)
        return(self._tri_fit_incr)



    @property
    def resid_us(self):
        """
        Compute and return unscaled Pearson residuals, given by
        r_us = (I - m)/sqrt(|m|), where r_us = unscaled Pearson residuals,
        I = Actual incremental loss and m = Fitted incremental loss.

        Returns
        -------
        pd.DataFrame
            DataFrame of unscaled Pearson residuals.
        """
        if self._resid_us is None:
            actuals = _cumtoincr(self.tri)
            fitted  = self.tri_fit_incr
            self._resid_us = (actuals - fitted) / np.sqrt(fitted.abs())
        return(self._resid_us)



    @property
    def resid_adj(self):
        """
        Compute and return the adjusted Pearson residuals, given by
        r_adj = sqrt(N/dof) * r_u, where r_adj = adjusted Pearson residuals,
        N = number of triangle cells, dof = degress of freedom and
        r_us = unscaled Pearson residuals.

        Returns
        -------
        pd.DataFrame
            DataFrame of adjusted Pearson residuals.
        """
        if self._resid_adj is None:
            self._resid_adj = \
                np.sqrt(self.nbr_cells / self.dof) * self.resid_us
        return(self._resid_adj)



    @property
    def sampling_dist(self):
        """
        Return self.resid_adj as a 1-dimensional array, which will be
        sampled from with replacement in order to produce synthetic
        triangles for bootstrapping. NaN's and 0's, if present in
        self.resid_adj, will not be present in self.sampling_dist.

        Returns
        -------
        np.ndarray
            Numpy array of adjusted residuals.
        """
        if self._sampling_dist is None:
            residuals = self.resid_adj.iloc[:-1,:-1].values.ravel()
            residuals = residuals[~np.isnan(residuals)]
            self._sampling_dist = residuals[residuals!=0]
        return(self._sampling_dist)




    def _bs_samples(self, sims=1000, parametric=False, random_state=None):
        """
        Return DataFrame containing sims resampled-with-replacement
        incremental loss triangles if ``parametric=False``, otherwise
        random variates from a normal distribution with mean and variance
        based on ``self.resid_adj``. Randomly generated incremental data
        will be cumulated in preparation for ldf calculation in next step.

        Parameters
        ----------
        sims: int
            The number of bootstrap simulations to run. Defaults to 1000.

        parametric: bool
            If True, fit standardized residuals to a normal distribution, and
            sample from the parameterized distribution. Otherwise, bootstrap
            procedure proceeds by sampling with replacement from the array
            of standardized residuals. Defaults to False.

        random_state: np.random.RandomState
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        pd.DataFrame
            DataFrame of bootstrapped synthetic loss data.
        """
        if random_state is not None:
            if isinstance(random_state, int):
                prng = RandomState(random_state)
            elif isinstance(random_state, RandomState):
                prng = random_state
        else:
            prng = RandomState()

        dfm = _tritotbl(self.tri_fit_incr)

        # Handle first period negative cells as specified by `neg_handler`.
        if np.any(dfm["value"] < 0):
            if self.neg_handler==1:
                dfm["value"] = np.where(
                    np.logical_and(dfm["dev"].values==1, dfm["value"].values<0), 1., dfm["value"].values
                    )
            elif self.neg_handler==2:
                # Bind reference to minimum triangle cell value.
                add2cells = np.abs(dfm["value"].min()) + 1
                dfm["value"] = dfm["value"] + add2cells
            else:
                raise ValueError("`neg_handler` must be in [1, 2].")

        dflist = [
            pd.DataFrame({"origin":i, "dev":self.tri.columns})
                for i in self.tri.index
            ]

        # Create single DataFrame from dflist.
        dfi = pd.DataFrame(np.vstack(dflist), columns=["origin", "dev"])
        dfp = dfi.merge(dfm, how="outer", on=["origin", "dev"])
        dfp["rectype"] = np.where(np.isnan(dfp["value"].values), "forecast", "actual")
        dfp.rename(columns={"value":"incr"}, inplace=True)
        dfp["incr_sqrt"] = np.sqrt(dfp["incr"].values)
        pretypes = dfp.dtypes.to_dict()

        # Replicate dfp self.sims times, then reset dtypes.
        dfr = pd.DataFrame(
            np.tile(dfp, (sims, 1)),
            columns=["origin", "dev", "incr", "rectype", "incr_sqrt"]
            )

        for col in dfr:
            dfr[col] = dfr[col].astype(pretypes[col])

        # Assign identifier to each record in dfr (`SIM`).
        dfr["sim"] = np.divmod(dfr.index, self.tri.shape[0] * self.tri.shape[1])[0]

        if parametric:
            # Set random residual on records in which rectype=="actual".
            _mm_mean, _mm_stdv = 0, self.sampling_dist.std(ddof=1, axis=0)
            dfr["resid"] = prng.normal(loc=_mm_mean, scale=_mm_stdv, size=dfr.shape[0])
        else:
            dfr["resid"] = prng.choice(self.sampling_dist, dfr.shape[0], replace=True)

        # Calcuate resampled incremental and cumulative losses.
        dfr["resid"] = np.where(dfr["rectype"].values=="forecast", np.NaN, dfr["resid"].values)
        dfr = dfr.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True)
        dfr["samp_incr"] = dfr["incr"].values + dfr["resid"].values * dfr["incr_sqrt"].values
        dfr["samp_cum"]  = dfr.groupby(["sim", "origin"])["samp_incr"].cumsum()
        return(dfr.reset_index(drop=True))




    def _bs_ldfs(self, samples_df):
        """
        Compute and return loss development factors for each set of
        synthetic loss data.

        Parameters
        ----------
        samples: pd.DataFrame
            Output from ``self._bs_samples`` method.

        Returns
        -------
        pd.DataFrame
        """
        keepcols = ["sim", "origin", "dev", "samp_cum", "last_origin"]
        lvi = self.tri.clvi.reset_index(drop=False)
        lvi.rename(
            columns={
                "index":"dev", "origin":"last_origin", "row_offset":"origin_offset"
                }, inplace=True
            )

        initdf = samples_df.merge(lvi, how="left", on=["dev"])
        initdf = initdf[keepcols].sort_values(by=["sim", "dev", "origin"])
        df = initdf[~np.isnan(initdf["samp_cum"])].reset_index(drop=True)
        df["_aggdev1"] = df.groupby(["sim", "dev"])["samp_cum"].transform("sum")
        df["_aggdev2"] = np.where(df["origin"].values==df["last_origin"].values, 0, df["samp_cum"].values)
        df["_aggdev2"] = df.groupby(["sim", "dev"])["_aggdev2"].transform("sum")
        uniqdf = df[["sim", "dev", "_aggdev1", "_aggdev2"]].drop_duplicates().reset_index(drop=True)
        uniqdf["_aggdev2"] = uniqdf["_aggdev2"].shift(periods=1, axis=0)
        uniqdf["dev"] = uniqdf["dev"].shift(periods=1, axis=0)
        ldfsdf        = uniqdf[uniqdf["_aggdev2"]!=0].dropna(how="any")
        ldfsdf["ldf"] = ldfsdf["_aggdev1"] / ldfsdf["_aggdev2"]
        ldfsdf["dev"] = ldfsdf["dev"].astype(np.integer)
        return(ldfsdf[["sim", "dev", "ldf"]].reset_index(drop=True))




    def _bs_forecasts(self, combined_df):
        """
        Populate lower-right of each simulated triangle using values from
        ``self._bs_samples`` and development factors from ``self._bs_ldfs``.

        Parameters
        ----------
        df: pd.DataFrame
            Combination of ``self._bs_samples``, ``self._bs_ldfs`` and
            ``self.tri.latest_by_origin``.

        Returns
        -------
        pd.DataFrame
            DataFrame of bootstrapped forecasts.
        """
        min_origin_year = combined_df["origin"].values.min()
        combined_df["_l_init_indx"] = np.where(combined_df["dev"].values>=combined_df["l_act_dev"].values, combined_df.index.values, -1)
        actsdf = combined_df[(combined_df["origin"].values==min_origin_year) | (combined_df["_l_init_indx"].values==-1)]
        fcstdf = combined_df[~combined_df.index.isin(actsdf.index)].sort_values(by=["sim", "origin", "dev"])
        fcstdf["_l_act_indx"] = fcstdf.groupby(["sim", "origin"])["_l_init_indx"].transform("min")
        fcstdf["l_act_cum"]   = fcstdf.lookup(fcstdf["_l_act_indx"].values, ["samp_cum"] * fcstdf.shape[0])
        fcstdf["_cum_ldf"]    = fcstdf.groupby(["sim", "origin"])["ldf"].transform("cumprod").shift(periods=1, axis=0)
        fcstdf["_samp_cum2"]  = np.nan_to_num((fcstdf["l_act_cum"].values * fcstdf["_cum_ldf"].values), 0)
        fcstdf["cum_final"]   = np.nan_to_num(fcstdf["samp_cum"].values, 0) + fcstdf["_samp_cum2"].values

        # Combine forecasts with actuals; compute incremental losses by sim and origin.
        fcstdf = fcstdf.drop(labels=["samp_cum", "samp_incr"], axis=1).rename(columns={"cum_final":"samp_cum"})
        sqrddf = pd.concat([fcstdf, actsdf], sort=True).sort_values(by=["sim", "origin", "dev"])
        sqrddf["_incr_dev1"] = np.nan_to_num(np.where(sqrddf["dev"].values==1, sqrddf["samp_cum"].values, np.NaN), 0)
        sqrddf["_incr_dev2"] = np.nan_to_num(sqrddf.groupby(["sim", "origin"])["samp_cum"].diff(periods=1), 0)
        sqrddf["samp_incr"]  = sqrddf["_incr_dev1"].values + sqrddf["_incr_dev2"].values
        sqrddf["var"]  = np.abs(sqrddf["samp_incr"].values * self.scale_param)
        sqrddf["sign"] = np.where(sqrddf["samp_incr"].values > 0, 1, -1)
        sqrddf.drop(labels=[i for i in sqrddf.columns if i.startswith("_")], axis=1, inplace=True)

        colorder = ["sim", "origin", "dev", "incr", "incr_sqrt", "rectype", "resid",
                    "samp_incr", "samp_cum", "ldf", "var", "sign", "l_act_dev",
                    "l_act_cum"]

        return(sqrddf[colorder].sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True))




    def _bs_process_error(self, forecasts_df, procdist="gamma", random_state=None):
        """
        Incorporate process variance by simulating each incremental future
        loss from ``procdist``. The mean is set to the forecast incremental
        loss amount and variance to `mean * self.scale_param`.
        The parameters for ``procdist`` must be positive. Since the mean
        and variance used to parameterize ``procdist`` depend on the
        resampled incremental losses, it is necessary to incorporate logic
        to address the possibility of negative incremental losses arising
        in the resampling stage. The approach used to handle negative
        incremental values is described in  Shapland[1], and replaces the
        distribution mean with the absolute value of the mean, and the
        variance to the absolute value of the mean multiplied by
        ``self.scale_param``.

        Parameters
        ----------
        forecasts: pd.DataFrame
            DateFrame of bootstraps forecasts generated within
            ``self._bs_forecasts``.

        procdist: str
            Specifies the distribution used to incorporate process error.
            Currently, can only be set to "gamma". Any other distribution
            will result in an error. Future release will also allow
            over-dispersed poisson ("odp"). If in the future ``procdist``
            is set to "odp", the negative binomial distribution is
            parameterized in such a way that results in a linear relationship
            between mean and variance.

        random_state: np.random.RandomState
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        pd.DataFrame
            DataFrame of cumulated, simulated losses after the incorporation
            of process variance.

        See Also
        --------
        ..[1] Shapland, Mark R - CAS Monograph Series Number 4:
              *Using the ODP Bootstrap Model: A Practicioner's Guide*,
              Casualty Actuarial Society, 2016.
        """
        # Initialize pseudo random number generator.
        if random_state is not None:
            if isinstance(random_state, int):
                prng = RandomState(random_state)
            elif isinstance(random_state, RandomState):
                prng = random_state
        else:
            prng = RandomState()

        # Parameterize distribution for process variance incorporation.
        if procdist.strip().lower()=="gamma":
            forecasts_df["param2"] = self.scale_param
            forecasts_df["param1"]  = np.abs(forecasts_df["samp_incr"].values / forecasts_df["param2"].values)

            def fdist(param1, param2):
                """gamma.rvs(a=param1, scale=param2, size=1, random_state=None)"""
                return(prng.gamma(param1, param2))
        else:
            raise ValueError(
                "Invalid procdist specification: `{}`".format(procdist)
                )

        forecasts_df["final_incr"] = np.where(
            forecasts_df["rectype"].values=="forecast",
            fdist(forecasts_df["param1"].values, forecasts_df["param2"].values) * forecasts_df["sign"].values,
            forecasts_df["samp_incr"].values
            )

        forecasts_df["final_cum"]  = forecasts_df.groupby(["sim", "origin"])["final_incr"].cumsum()

        return(forecasts_df.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True))




    @staticmethod
    def _bs_reserves(process_error_df):
        """
        Compute unpaid loss reserve estimate using output from
        ``self._bs_process_error``.

        Parameters
        ----------
        pedf: pd.DataFrame
            Output from ``self._bs_process_error``.

        Returns
        -------
        pd.DataFrame
            DataFrame with unpaid/ibnr estimate.
        """
        keepcols = ["sim", "origin", "latest", "ultimate", "reserve"]
        max_devp = process_error_df["dev"].values.max()
        process_error_df = process_error_df.rename(columns={"final_cum":"ultimate", "l_act_cum":"latest"})
        process_error_df["reserve"] = process_error_df["ultimate"] - process_error_df["latest"]
        resvdf = process_error_df[process_error_df["dev"].values==max_devp][keepcols].drop_duplicates()
        resvdf["latest"]  = np.where(np.isnan(resvdf["latest"].values), resvdf["ultimate"].values, resvdf["latest"].values)
        resvdf["reserve"] = np.nan_to_num(resvdf["reserve"].values, 0)
        return(resvdf.sort_values(by=["origin", "sim"]).reset_index(drop=True))




    @property
    def fit_assessment(self):
        """
        Return a statistical summary assessing the fit of the parametric
        model used for bootstrap resampling (applicable when ``parametric``
        argument to __call__  is True).

        Returns
        -------
        dict
            Dictionary with keys ``kstest``, ``anderson``, ``shapiro``,
            ``skewtest``, ``kurtosistest`` and ``normaltest``,
            corresponding to statistical tests available in ``scipy.stats``.
        """
        if self._fit_assessment is None:
            _mm_mean = self.sampling_dist.mean()
            _mm_stdv = self.sampling_dist.std(ddof=1, axis=0)
            _mm_dist = stats.norm(loc=_mm_mean, scale=_mm_stdv)
            D, p_ks  = stats.kstest(self.sampling_dist, _mm_dist.cdf)
            W, p_sw  = stats.shapiro(self.sampling_dist)
            Z, p_sk  = stats.skewtest(self.sampling_dist, axis=0, nan_policy="omit")
            K, p_kt  = stats.kurtosistest(self.sampling_dist, axis=0, nan_policy="omit")
            S, p_nt  = stats.normaltest(self.sampling_dist, axis=0, nan_policy="omit")
            A, crit, sig = stats.anderson(self.sampling_dist, dist="norm")
            self._fit_assessment = {
                "kstest"      :{"statistic":D, "p-value":p_ks},
                "anderson"    :{"statistic":A, "critical_values":crit, "significance_levels":sig},
                "shapiro"     :{"statistic":W, "p-value":p_sw},
                "skewtest"    :{"statistic":Z, "p-value":p_sk},
                "kurtosistest":{"statistic":K, "p-value":p_kt},
                "normaltest"  :{"statistic":S, "p-value":p_nt},
                }
        return(self._fit_assessment)



    @property
    def residuals_detail(self):
        """
        Compute summary statistics based on triangle residuals.

        Returns
        -------
        pd.DataFrame
            DataFrame consisting of unscaled and adjusted residual summary
            statistics.
        """
        if self._residuals_detail is None:
            _unscaled      = self.resid_us.values.ravel()
            _adjusted      = self.resid_adj.values.ravel()
            _unscaled      = _unscaled[~np.isnan(_unscaled)]
            _adjusted      = _adjusted[~np.isnan(_adjusted)]
            _unscaled      = _unscaled[_unscaled!=0]
            _adjusted      = _adjusted[_adjusted!=0]
            _unscaled_size = _unscaled.size
            _unscaled_sum  = _unscaled.sum(axis=0)
            _unscaled_ssqr = np.sum(_unscaled**2, axis=0)
            _unscaled_min  = _unscaled.min(axis=0)
            _unscaled_max  = _unscaled.max(axis=0)
            _unscaled_mean = _unscaled.mean(axis=0)
            _unscaled_skew = stats.skew(_unscaled, axis=0, nan_policy="omit")
            _unscaled_mode = stats.mode(_unscaled, axis=0, nan_policy="omit").mode[0]
            _unscaled_cvar = stats.variation(_unscaled, axis=0, nan_policy="omit")
            _unscaled_kurt = stats.kurtosis(_unscaled, axis=0, nan_policy="omit")
            _unscaled_var  = _unscaled.var(ddof=1, axis=0)
            _unscaled_stdv = _unscaled.std(ddof=1, axis=0)
            _unscaled_med  = np.median(_unscaled, axis=0)
            _adjusted_size = _adjusted.size
            _adjusted_sum  = _adjusted.sum(axis=0)
            _adjusted_ssqr = np.sum(_adjusted**2, axis=0)
            _adjusted_min  = _adjusted.min(axis=0)
            _adjusted_max  = _adjusted.max(axis=0)
            _adjusted_mean = _adjusted.mean(axis=0)
            _adjusted_skew = stats.skew(_adjusted, axis=0, nan_policy="omit")
            _adjusted_mode = stats.mode(_adjusted, axis=0, nan_policy="omit").mode[0]
            _adjusted_cvar = stats.variation(_adjusted, axis=0, nan_policy="omit")
            _adjusted_kurt = stats.kurtosis(_adjusted, axis=0, nan_policy="omit")
            _adjusted_var  = _adjusted.var(ddof=1, axis=0)
            _adjusted_stdv = _adjusted.std(ddof=1, axis=0)
            _adjusted_med  = np.median(_adjusted, axis=0)
            self._residuals_detail = pd.DataFrame({
                "unscaled":[
                    _unscaled_size, _unscaled_sum , _unscaled_ssqr, _unscaled_min,
                    _unscaled_max,  _unscaled_mean, _unscaled_skew, _unscaled_mode,
                    _unscaled_cvar, _unscaled_kurt, _unscaled_var , _unscaled_stdv,
                    _unscaled_med,
                    ],
                "adjusted":[
                    _adjusted_size, _adjusted_sum , _adjusted_ssqr, _adjusted_min,
                    _adjusted_max,  _adjusted_mean, _adjusted_skew, _adjusted_mode,
                    _adjusted_cvar, _adjusted_kurt, _adjusted_var , _adjusted_stdv,
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








class _BootstrapResult:
    """
    Curated output generated from ``_BootstrapChainLadder``'s __call__ method.
    """
    def __init__(self, summary_df, reserves_df, process_error_df, **kwargs):

        self.summary = summary_df
        self.reserves = reserves_df
        self.prerror = process_error_df

        if kwargs is not None:
            for key_ in kwargs:
                setattr(self, key_, kwargs[key_])

        # Properties
        self._origindist = None
        self._aggdist = None


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
            Number of bins to use for histogram representation.
        """
        dat = np.array(data, dtype=np.float_)
        IQR = stats.iqr(dat, rng=(25, 75), scale="raw", nan_policy="omit")
        N = dat.size
        bw = (2 * IQR) / np.power(N, 1/3)
        datmin, datmax = dat.min(), dat.max()
        datrng = datmax - datmin
        return(int((datrng / bw) + 1))



    @property
    def aggdist(self):
        """
        Return aggregate distribution of simulated reserve amounts
        over all origin years.
        """
        if self._aggdist is None:
            self._aggdist = self.reserves.groupby(
                ["sim"], as_index=False)[["latest", "ultimate", "reserve"]].sum()
        return(self._aggdist)



    @property
    def origindist(self):
        """
        Return distribution of simulated loss reserves by origin year.
        """
        if self._origindist is None:
            self._origindist = self.reserves.groupby(
                ["sim", "origin"], as_index=False)[["latest", "ultimate", "reserve"]].sum()
        return(self._origindist)



    def plotdist(self, level="aggregate", tc="#FE0000", path=None, **kwargs):
        """
        Generate visual representation of full predicitive distribtion
        of loss reserves in aggregate or by origin. Additional options
        to style the histogram can be passed as keyword arguments.

        Parameters
        ----------
        level: str
            Set ``level`` to "origin" for faceted plot with predicitive
            distirbution by origin year, or "aggregate" for single plot
            of predicitive distribution of reserves in aggregate. Default
            value is "aggregate".

        path: str
            If path is not None, save plot to the specified location.
            Otherwise, parameter is ignored. Default value is None.

        kwargs: dict
            Dictionary of optional matplotlib styling parameters.

        """
        plt_params = {
            "alpha":.995, "color":"#FFFFFF", "align":"mid", "edgecolor":"black",
            "histtype":"bar", "linewidth":1.1, "orientation":"vertical",
            }

        if level.lower().strip().startswith(("agg", "tot")):
            # bins computed using self._nbrbins if not passed as optional
            # keyword argument.
            dat = self.aggdist["reserve"].values
            plt_params["bins"] = self._nbrbins(data=dat)

            # Update plt_params with any optional keyword arguments.
            plt_params.update(kwargs)

            # Setup.
            fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True)
            ax.set_facecolor("#1f77b4")
            ax.set_title(
                "Distribution of Bootstrap Reserve Estimates (Aggregate)",
                loc="left", color=tc)
            ax.get_xaxis().set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.set_xlabel("Reserves"); ax.set_ylabel("Frequency")
            ax.hist(dat, **plt_params)

        elif level.lower().strip().startswith(("orig", "year")):
            dat = self.origindist[["origin", "reserve"]]
            sns.set(rc={'axes.facecolor':"#1f77b4"})
            g = sns.FacetGrid(dat, col="origin", col_wrap=4, margin_titles=False)
            g.map(plt.hist, "reserve", **plt_params)
            g.set_titles("{col_name}", color=tc)
            g.fig.suptitle("Reserve Distribution by Origin Year", color=tc, weight="bold")
            plt.subplots_adjust(top=0.92)
        plt.show()
