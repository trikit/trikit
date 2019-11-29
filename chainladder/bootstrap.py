"""
BootstrapChainLadder Implementation.

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

- Implement heteroscedasticity adjustment (Shapland pg. 25)
- Plot residuals vs. development period
- Plot residuals vs. origin
- Separate class for residuals assessment


"""
import functools
import warnings
import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from ..chainladder import BaseChainLadder, BaseChainLadderResult







class BootstrapChainLadder(BaseChainLadder):
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
    to avoid the complication associated with modeling a tail factor.

    References
    ----------
    - England, P., and R. Verrall, (2002), *Stochastic Claims Reserving in General
    Insurance*, British Actuarial Journal 8(3): 443-518.
    - CAS Working Party on Quantifying Variability in Reserve Estimates,
    *The Analysis and Estimation of Loss & ALAE Variability: A Summary Report*,
    Casualty Actuarial Society Forum, Fall 2005.
    - Leong et al., (2012), *Back-Testing the ODP Bootstrap of the Paid
    Chain-Ladder Model with Actual Historical Claims Data*, Casualty Actuarial
    Society E-Forum.
    - Kirschner, et al., *Two Approaches to Calculating Correlated Reserve
    Indications Across Multiple Lines of Business* Appendix III, Variance
    Journal, Volume 2/Issue 1.
    - Shapland, Mark R., (2016), *Using the ODP Bootstrap Model: A
    Practicioner's Guide*, CAS Monograph Series Number 4: Casualty Actuarial
    Society, 2016.
    """
    def __init__(self, cumtri):
        """
        The BootstrapChainLadder class definition.

        Parameters
        ----------
        cumtri: triangle._CumTriangle
            A cumulative.CumTriangle instance.
        """
        super().__init__(cumtri=cumtri)



    def __call__(self, sims=1000, q=[.75, .95], neg_handler=1, procdist="gamma",
                 parametric=False, symmetric=False, interpolation="linear",
                 random_state=None):
        """
        ``BootstrapChainLadder`` simulation initializer. Generates predictive
        distribution of ultimate reserve outcomes.

        As stated in ``BootstrapChainLadder``'s documentation, the estimated
        distribution of losses assumes development is complete by the final
        development period in order to avoid the complication of modeling a
        tail factor.

        Parameters
        ----------
        sims: int
            The number of bootstrap simulations to perfrom. Defaults to 1000.

        q: float in range of [0,1] (or sequence of floats)
            Percentile to compute, which must be between 0 and 1 inclusive.

        neg_handler: int
            If ``neg_handler=1``, then any first development period negative
            cells will be coerced to +1. If ``neg_handler=2``, the minimum
            value in all triangle cells is identified (identified as 'MIN_CELL').
            If MIN_CELL is less than or equal to 0, the equation
            (MIN_CELL + X = +1.0) is solved for X. X is then added to every
            other cell in the triangle, resulting in all triangle cells having
            a value strictly greater than 0.

        procdist: str
            The distribution used to incorporate process variance. Currently,
            this can only be set to "gamma".

        symmetric: bool
            Whether the symmetric interval should be included in summary
            output. For example, if ``symmetric==True`` and ``q=.95``, then
            the 2.5th and 97.5th quantiles of the bootstrapped reserve
            distribution will be returned [(1 - .95) / 2, (1 + .95) / 2]. When
            False, only the specified quantile(s) will be computed. Defaults
            to False.

        parametric: bool
            If True, fit standardized residuals to a normal distribution, and
            sample from this parameterized distribution. Otherwise, bootstrap
            procedure samples with replacement from the collection of
            standardized residuals. Defaults to False.

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
        _BootstrapChainLadderResult
        """
        # Obtain reference to Chain ladder point estimates.
        ldfs_ = self._ldfs(sel="all-weighted")
        cldfs_ = self._cldfs(ldfs=ldfs_)
        ultimates_ = self._ultimates(cldfs=cldfs_)
        reserves_ = self._reserves(ultimates=ultimates_)
        maturity_ = self.tri.maturity.astype(np.str)
        latest_ = self.tri.latest_by_origin
        trisqrd_ = self._trisqrd(ldfs=ldfs_)

        # Obtain reference to Bootstrap estimates.
        tri_fit_cum_ = self._tri_fit_cum(sel="all-weighted")
        tri_fit_incr_ = self._tri_fit_incr(fitted_tri_cum=tri_fit_cum_)
        unscld_residuals_ = self._resid_us(fitted_tri_incr=tri_fit_incr_)
        adjust_residuals_ = self._resid_adj(resid_us=unscld_residuals_)
        scale_param_ = self._scale_param(resid_us=unscld_residuals_)
        sampling_dist_ = self._sampling_dist(resid_adj=adjust_residuals_)
        dfsamples = self._bs_samples(
            sampling_dist=sampling_dist_, fitted_tri_incr=tri_fit_incr_,
            sims=sims, neg_handler=neg_handler, parametric=parametric,
            random_state=random_state
            )

        dfldfs = self._bs_ldfs(dfsamples=dfsamples)
        dflvi = self.tri.rlvi.reset_index(drop=False)
        dflvi = dflvi.rename({"index":"origin", "dev":"l_act_dev"}, axis=1)
        dflvi = dflvi.drop("col_offset", axis=1)
        dfcombined = dfsamples.merge(dfldfs, on=["sim", "dev"], how="left")
        dfcombined = dfcombined.merge(dflvi, on=["origin"], how="left", )
        dfcombined = dfcombined.reset_index(drop=True).sort_values(by=["sim", "origin", "dev"])
        dfforecasts = self._bs_forecasts(dfcombined=dfcombined, scale_param=scale_param_)
        dfprocerror = self._bs_process_error(
            dfforecasts=dfforecasts, scale_param=scale_param_, procdist=procdist,
            random_state=random_state)
        dfreserves = self._bs_reserves(dfprocerror=dfprocerror)
        pctl_ = np.asarray([q] if isinstance(q, (float, int)) else q)

        if np.any(np.logical_or(pctl_ <= 1, pctl_ >= 0)):
            if symmetric:
                pctlarr = np.sort(np.unique(np.append((1 - pctl_) / 2, (1 + pctl_) / 2)))
            else:
                pctlarr = np.sort(np.unique(pctl_))
        else:
            raise ValueError("Values for quantiles must fall between [0, 1].")

        # Compile Chain Ladder point estimate summary.
        dfmatur_ = maturity_.to_frame().reset_index(drop=False).rename({"index":"origin"}, axis=1)
        dfcldfs_ = cldfs_.to_frame().reset_index(drop=False).rename({"index":"maturity"}, axis=1)
        dfcldfs_["maturity"] = dfcldfs_["maturity"].astype(np.str)
        dfsumm = dfmatur_.merge(dfcldfs_, on=["maturity"], how="left").set_index("origin")
        dfsumm.index.name = None
        dflatest_ = latest_.to_frame().rename({"latest_by_origin":"latest"}, axis=1)
        dfultimates_ = ultimates_.to_frame()
        dfreserves_ = reserves_.to_frame()
        dfsumm = functools.reduce(
            lambda df1, df2: df1.join(df2),
            (dflatest_, dfultimates_, dfreserves_), dfsumm
            )

        # Attach percentile fields to dfsumm.
        pctlfmt = ["{:.5f}".format(i).rstrip("0").rstrip(".") + "%" for i in 100 * pctlarr]
        for q_, pctlstr_ in zip(pctlarr, pctlfmt):
            dfsumm[pctlstr_] = dfsumm.index.map(
                lambda v: np.percentile(
                    dfreserves[dfreserves["origin"]==v]["reserve"].values,
                    100 * q_, interpolation=interpolation
                    )
                )

        # Add "Total" index and set to NaN fields that shouldn't be aggregated.
        dfsumm.loc["total"] = dfsumm.sum()
        dfsumm.loc["total", "maturity"] = ""
        dfsumm.loc["total", "cldf"] = np.NaN
        dfsumm = dfsumm.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        kwds = {"sel":"all-weighted", "sims": sims, "neg_handler":neg_handler,
                "procdist":procdist, "parametric":parametric,
                "q":q, "interpolation":interpolation,}
        sampling_dist_res = None if parametric==True else sampling_dist_
        clresult_ = BootstrapChainLadderResult(
            summary=dfsumm, reserve_dist=dfreserves, sims_data=dfprocerror,
            tri=self.tri, ldfs=ldfs_, cldfs=cldfs_, latest=latest_,
            maturity=maturity_, ultimates=ultimates_, reserves=reserves_,
            scale_param=scale_param_, unscaled_residuals=unscld_residuals_,
            adjusted_residuals=adjust_residuals_, sampling_dist=sampling_dist_res,
            fitted_tri_cum=tri_fit_cum_, fitted_tri_incr=tri_fit_incr_,
            trisqrd=trisqrd_, **kwds
            )
        return(clresult_)


    def _scale_param(self, resid_us):
        """
        Return the scale parameter, which is the sum of the squared unscaled
        Pearson residuals over the degrees of freedom. This method is intended
        for internal use only.

        Parameters
        ----------
        resid_us: pd.DataFrame
            Unscaled Pearson residuals, typically output by
            ``self._resid_us``.

        Returns
        -------
        float
        """
        return((resid_us**2).sum().sum() / self.tri.dof)


    def _tri_fit_cum(self, sel="all-weighted"):
        """
        Return the cumulative fitted triangle using backwards recursion,
        starting with the observed cumulative paid/incurred-to-date along the
        latest diagonal.

        Parameters
        ----------
        sel: str
            The ldf average to select from ``triangle._CumTriangle.a2a_avgs``.
            Defaults to "all-weighted".

        Returns
        -------
        pd.DataFrame
        """
        ldfs_ = self._ldfs(sel=sel)
        fitted_tri_cum_ = self.tri.copy(deep=True)
        for i in range(fitted_tri_cum_.shape[0]):
            iterrow = fitted_tri_cum_.iloc[i, :]
            if iterrow.isnull().any():
                # Find first NaN element in iterrow.
                nan_hdr = iterrow.isnull()[iterrow.isnull()==True].index[0]
                nan_idx = fitted_tri_cum_.columns.tolist().index(nan_hdr)
                init_idx = nan_idx - 1
            else:
                # If here, iterrow is the most mature exposure period.
                init_idx = fitted_tri_cum_.shape[1] - 1
            # Set to NaN any development periods earlier than init_idx.
            fitted_tri_cum_.iloc[i, :init_idx] = np.NaN
            # Iterate over rows, undeveloping triangle from latest diagonal.
            for j in range(fitted_tri_cum_.iloc[i, :init_idx].size, 0, -1):
                prev_col_idx, curr_col_idx, curr_ldf_idx = j, j - 1, j - 1
                prev_col_val = fitted_tri_cum_.iloc[i, prev_col_idx]
                curr_ldf_val = ldfs_.iloc[curr_ldf_idx]
                fitted_tri_cum_.iloc[i, curr_col_idx] = (prev_col_val / curr_ldf_val)
        return(fitted_tri_cum_)


    @staticmethod
    def _tri_fit_incr(fitted_tri_cum):
        """
        Return the fitted incremental triangle.

        Parameters
        ----------
        fitted_tri_cum: pd.DataFrame
            Typically the output from ``self._tri_fit_cum``.

        Returns
        -------
        pd.DataFrame
        """
        tri_ = fitted_tri_cum.diff(axis=1)
        tri_.iloc[:, 0] = fitted_tri_cum.iloc[:, 0]
        return(tri_)



    def _resid_us(self, fitted_tri_incr):
        """
        Return unscaled Pearson residuals, given by
        $r_{us} = \frac{I - m}{\sqrt{|m|}}$, where $r_{us}$ represents the
        unscaled Pearson residuals, $I$ the actual incremental losses and $m$
        the fitted incremental losses.

        Parameters
        ----------
        fitted_tri_incr: pd.DataFrame
            Typically the output from ``self._tri_fit_incr``.

        Returns
        -------
        pd.DataFrame
        """
        I_ = self.tri.as_incr() # Actual incremental losses
        m_ = fitted_tri_incr    # Fitted incremental losses
        return((I_ - m_) / np.sqrt(m_.abs()))


    def _resid_adj(self, resid_us):
        """
        Compute and return the adjusted Pearson residuals, given by
        $r_{adj} = \sqrt{\frac{N}{dof}} * r_{us}$, where $r_adj$ represents
        the adjusted Pearson residuals, $N$ the number of triangle cells,
        $dof$ the degress of freedom and $r_{us}$ the unscaled Pearson
        residuals.

        Parameters
        ----------
        resid_us: pd.DataFrame
            Unscaled Pearson residuals, typically output by
            ``self._resid_us``.

        Returns
        -------
        pd.DataFrame
        """
        return(np.sqrt(self.tri.nbr_cells / self.tri.dof) * resid_us)


    @staticmethod
    def _sampling_dist(resid_adj):
        """
        Return ``resid_adj`` as a 1-dimensional array, which will be sampled
        from with replacement in order to produce synthetic triangles for
        bootstrapping. Any NaN's and 0's present in ``resid_adj`` will not be
        present in the returned array.

        Parameters
        ----------
        resid_adj: pd.DataFrame
            Adjusted Pearson residuals, typically output by
            ``self._resid_adj``.

        Returns
        -------
        np.ndarray
        """
        resid_ = resid_adj.iloc[:-1,:-1].values.ravel()
        return(resid_[np.logical_and(~np.isnan(resid_), resid_!=0)])


    def _bs_samples(self, sampling_dist, fitted_tri_incr, sims=1000,
                    neg_handler=1, parametric=False, random_state=None):
        """
        Return DataFrame containing sims resampled-with-replacement
        incremental loss triangles if ``parametric=False``, otherwise
        random variates from a normal distribution with mean zero and
        variance derived from ``resid_adj``. Randomly generated incremental
        data gets cumulated in preparation for ldf calculation in next
        step.

        Parameters
        ----------
        sampling_dist: np.ndarray
            The residuals from the fitted incremental triangle coerced
            into a one-dimensional numpy array.

        fitted_tri_incr: pd.DataFrame
            The incremental triangle fitted using backwards recursion.
            Typically the output of ``self._tri_fit_incr``.

        sims: int
            The number of bootstrap simulations to run. Defaults to 1000.

        neg_handler: int
            If ``neg_handler=1``, then any first development period negative
            cells will be coerced to +1. If ``neg_handler=2``, the minimum
            value in all triangle cells is identified (identified as 'MIN_CELL').
            If MIN_CELL is less than or equal to 0, the equation
            (MIN_CELL + X = +1.0) is solved for X. X is then added to every
            other cell in the triangle, resulting in all triangle cells having
            a value strictly greater than 0.

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
        """
        if random_state is not None:
            if isinstance(random_state, int):
                prng = RandomState(random_state)
            elif isinstance(random_state, RandomState):
                prng = random_state
        else:
            prng = RandomState()

        sampling_dist_ = sampling_dist.ravel()

        fti_ = fitted_tri_incr.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        dfm = pd.melt(fti_, id_vars=["origin"], var_name="dev", value_name="value")
        dfm = dfm[~np.isnan(dfm["value"])].astype(
            {"origin":np.int_, "dev":np.int_, "value":np.float_})

        # Handle first period negative cells as specified by `neg_handler`.
        if np.any(dfm["value"] < 0):
            if neg_handler==1:
                dfm["value"] = np.where(
                    np.logical_and(dfm["dev"].values==1, dfm["value"].values<0),
                    1., dfm["value"].values
                    )

            elif neg_handler==2:
                # Obtain reference to minimum triangle cell value, then
                # add the absolute value of that amount plus one to every
                # other triangle cell.
                add2cells = np.abs(dfm["value"].min()) + 1
                dfm["value"] = dfm["value"] + add2cells
            else:
                raise ValueError("`neg_handler` must be in [1, 2].")

        # Note that we don't use self.tri's ``as_tbl`` method since we need
        # to retain records with NaNs.
        dftri = self.tri.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        dfi = pd.melt(dftri, id_vars=["origin"], var_name="dev", value_name="value").drop("value", axis=1)
        dfp = dfi.merge(dfm, how="outer", on=["origin", "dev"])
        dfp["rectype"] = np.where(np.isnan(dfp["value"].values), "forecast", "actual")
        dfp = dfp.rename({"value":"incr"}, axis=1)
        dfp["incr_sqrt"] = np.sqrt(dfp["incr"].values)
        dtypes_ = {"origin":np.int, "dev":np.int_, "incr":np.float_,
                   "incr_sqrt":np.float_, "rectype":np.str,}

        # Replicate dfp sims times then redefine datatypes.
        dfr = pd.DataFrame(
            np.tile(dfp, (sims, 1)),
            columns=["origin", "dev", "incr", "rectype", "incr_sqrt"],
            )

        for hdr_ in dfr.columns:
            dfr[hdr_] = dfr[hdr_].astype(dtypes_[hdr_])

        # Assign simulation identifier to each record in dfr.
        dfr["sim"] = np.divmod(dfr.index, self.tri.shape[0] * self.tri.shape[1])[0]
        sample_size_ = dfr.shape[0]

        if parametric:
            # Sample random residuals from normal distribution with zero mean.
            stddev_ = sampling_dist_.std(ddof=1)
            dfr["resid"] = prng.normal(loc=0, scale=stddev_, size=sample_size_)
        else:
            # Sample random residual from adjusted pearson residuals.
            dfr["resid"] = prng.choice(sampling_dist_, sample_size_, replace=True)

        # Calcuate simulated incremental and cumulative losses.
        dfr["resid"] = np.where(dfr["rectype"].values=="forecast", np.NaN, dfr["resid"].values)
        dfr = dfr.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True)
        dfr["samp_incr"] = dfr["incr"].values + dfr["resid"].values * dfr["incr_sqrt"].values
        dfr["samp_cum"]  = dfr.groupby(["sim", "origin"])["samp_incr"].cumsum()
        return(dfr.reset_index(drop=True))


    def _bs_ldfs(self, dfsamples):
        """
        Compute and return loss development factors for each set of
        synthetic loss data. This method is intended for internal use
        only.

        Parameters
        ----------
        dfsamples: pd.DataFrame
            Output from ``self._bs_samples``.

        Returns
        -------
        pd.DataFrame
        """
        keepcols = ["sim", "origin", "dev", "samp_cum", "last_origin"]
        dflvi = self.tri.clvi.reset_index(drop=False)
        dflvi = dflvi.rename(
            {"index":"dev", "origin":"last_origin", "row_offset":"origin_offset"}, axis=1)
        dfinit = dfsamples.merge(dflvi, how="left", on=["dev"])
        dfinit = dfinit[keepcols].sort_values(by=["sim", "dev", "origin"])
        df = dfinit[~np.isnan(dfinit["samp_cum"])].reset_index(drop=True)
        df["_aggdev1"] = df.groupby(["sim", "dev"])["samp_cum"].transform("sum")
        df["_aggdev2"] = np.where(df["origin"].values==df["last_origin"].values, 0, df["samp_cum"].values)
        df["_aggdev2"] = df.groupby(["sim", "dev"])["_aggdev2"].transform("sum")
        dfuniq = df[["sim", "dev", "_aggdev1", "_aggdev2"]].drop_duplicates().reset_index(drop=True)
        dfuniq["_aggdev2"] = dfuniq["_aggdev2"].shift(periods=1)
        dfuniq["dev"] = dfuniq["dev"].shift(periods=1)
        dfldfs = dfuniq[dfuniq["_aggdev2"]!=0].dropna(how="any")
        dfldfs["ldf"] = dfldfs["_aggdev1"] / dfldfs["_aggdev2"]
        dfldfs["dev"] = dfldfs["dev"].astype(np.int_)
        return(dfldfs[["sim", "dev", "ldf"]].reset_index(drop=True))


    @staticmethod
    def _bs_forecasts(dfcombined, scale_param):
        """
        Populate lower-right of each simulated triangle using values from
        ``self._bs_samples`` and development factors from ``self._bs_ldfs``.

        Parameters
        ----------
        dfcombined: pd.DataFrame
            Combination of ``self._bs_samples``, ``self._bs_ldfs`` and
            ``self.tri.latest_by_origin``.

        scale_param: float
            the sum of the squared unscaled Pearson residuals over the
            degrees of freedom. Computed within ``self._scale_param``.

        Returns
        -------
        pd.DataFrame
        """
        min_origin_year_ = dfcombined["origin"].values.min()
        dfcombined["_l_init_indx"] = np.where(
            dfcombined["dev"].values>=dfcombined["l_act_dev"].values, dfcombined.index.values, -1)
        dfacts = dfcombined[(dfcombined["origin"].values==min_origin_year_) | (dfcombined["_l_init_indx"].values==-1)]
        dffcst = dfcombined[~dfcombined.index.isin(dfacts.index)].sort_values(by=["sim", "origin", "dev"])
        dffcst["_l_act_indx"] = dffcst.groupby(["sim", "origin"])["_l_init_indx"].transform("min")
        dffcst["l_act_cum"] = dffcst.lookup(dffcst["_l_act_indx"].values, ["samp_cum"] * dffcst.shape[0])
        dffcst["_cum_ldf"] = dffcst.groupby(["sim", "origin"])["ldf"].transform("cumprod").shift(periods=1, axis=0)
        dffcst["_samp_cum2"] = np.nan_to_num((dffcst["l_act_cum"].values * dffcst["_cum_ldf"].values), 0)
        dffcst["cum_final"] = np.nan_to_num(dffcst["samp_cum"].values, 0) + dffcst["_samp_cum2"].values

        # Combine forecasts with actuals then compute incremental losses by sim and origin.
        dffcst = dffcst.drop(labels=["samp_cum", "samp_incr"], axis=1).rename(columns={"cum_final":"samp_cum"})
        dfsqrd = pd.concat([dffcst, dfacts], sort=True).sort_values(by=["sim", "origin", "dev"])
        dfsqrd["_incr_dev1"] = np.nan_to_num(np.where(dfsqrd["dev"].values==1, dfsqrd["samp_cum"].values, np.NaN), 0)
        dfsqrd["_incr_dev2"] = np.nan_to_num(dfsqrd.groupby(["sim", "origin"])["samp_cum"].diff(periods=1), 0)
        dfsqrd["samp_incr"] = dfsqrd["_incr_dev1"].values + dfsqrd["_incr_dev2"].values
        dfsqrd["var"] = np.abs(dfsqrd["samp_incr"].values * scale_param)
        dfsqrd["sign"] = np.where(dfsqrd["samp_incr"].values > 0, 1, -1)
        dfsqrd = dfsqrd.drop(labels=[i for i in dfsqrd.columns if i.startswith("_")], axis=1)
        return(dfsqrd.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True))


    @staticmethod
    def _bs_process_error(dfforecasts, scale_param, procdist="gamma", random_state=None):
        """
        Incorporate process error by simulating each incremental future
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

        scale_param: float
            the sum of the squared unscaled Pearson residuals over the
            degrees of freedom. Computed within ``self._scale_param``.

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
            dfforecasts["param2"] = scale_param
            dfforecasts["param1"] = np.abs(dfforecasts["samp_incr"].values / dfforecasts["param2"].values)
            def fdist(param1, param2):
                """gamma.rvs(a=param1, scale=param2, size=1, random_state=None)"""
                return(prng.gamma(param1, param2))
        else:
            raise ValueError("Invalid procdist specification: `{}`".format(procdist))
        dfforecasts["final_incr"] = np.where(
            dfforecasts["rectype"].values=="forecast",
            fdist(dfforecasts["param1"].values, dfforecasts["param2"].values) * dfforecasts["sign"].values,
            dfforecasts["samp_incr"].values)
        dfforecasts["final_cum"] = dfforecasts.groupby(["sim", "origin"])["final_incr"].cumsum()
        dfforecasts = dfforecasts.rename({"final_cum":"ultimate", "l_act_cum":"latest"}, axis=1)
        return(dfforecasts.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True))


    @staticmethod
    def _bs_reserves(dfprocerror):
        """
        Compute unpaid loss reserve estimate using output from
        ``self._bs_process_error``.

        Parameters
        ----------
        dfprocerror: pd.DataFrame
            Output from ``self._bs_process_error``.

        Returns
        -------
        pd.DataFrame
        """
        keepcols_ = ["sim", "origin", "latest", "ultimate", "reserve"]
        max_devp_ = dfprocerror["dev"].values.max()
        dfprocerror["reserve"] = dfprocerror["ultimate"] - dfprocerror["latest"]
        dfreserves_ = dfprocerror[dfprocerror["dev"].values==max_devp_][keepcols_].drop_duplicates()
        dfreserves_["latest"]  = np.where(
            np.isnan(dfreserves_["latest"].values), dfreserves_["ultimate"].values, dfreserves_["latest"].values)
        dfreserves_["reserve"] = np.nan_to_num(dfreserves_["reserve"].values, 0)
        return(dfreserves_.sort_values(by=["origin", "sim"]).reset_index(drop=True))





class BootstrapChainLadderResult(BaseChainLadderResult):
    """
    Curated output resulting from ``BootstrapChainLadder``'s ``run`` method.
    """
    def __init__(self, summary, reserve_dist, sims_data, tri, ldfs, cldfs,
                 latest, maturity, ultimates, reserves, scale_param,
                 unscaled_residuals, adjusted_residuals, sampling_dist,
                 fitted_tri_cum, fitted_tri_incr, trisqrd, **kwargs):
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
            keepcols_ = ["latest", "ultimate", "reserve"]
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
                unscaled_ = self.unscaled_residuals.values.ravel()
                adjusted_ = self.adjusted_residuals.values.ravel()
                unscaled_ = unscaled_[~np.isnan(unscaled_)]
                adjusted_ = adjusted_[~np.isnan(adjusted_)]
                unscaled_ = unscaled_[unscaled_!=0]
                adjusted_ = adjusted_[adjusted_!=0]
                unscaled_size_ = unscaled_.size
                unscaled_sum_ = unscaled_.sum(axis=0)
                unscaled_ssqr_ = np.sum(unscaled_**2, axis=0)
                unscaled_min_  = unscaled_.min(axis=0)
                unscaled_max_  = unscaled_.max(axis=0)
                unscaled_mean_ = unscaled_.mean(axis=0)
                unscaled_skew_ = stats.skew(unscaled_, axis=0, nan_policy="omit")
                unscaled_mode_ = stats.mode(unscaled_, axis=0, nan_policy="omit").mode[0]
                unscaled_cvar_ = stats.variation(unscaled_, axis=0, nan_policy="omit")
                unscaled_kurt_ = stats.kurtosis(unscaled_, axis=0, nan_policy="omit")
                unscaled_var_  = unscaled_.var(ddof=1, axis=0)
                unscaled_stddev_ = unscaled_.std(ddof=1, axis=0)
                unscaled_med_  = np.median(unscaled_, axis=0)
                adjusted_size_ = adjusted_.size
                adjusted_sum_  = adjusted_.sum(axis=0)
                adjusted_ssqr_ = np.sum(adjusted_**2, axis=0)
                adjusted_min_  = adjusted_.min(axis=0)
                adjusted_max_  = adjusted_.max(axis=0)
                adjusted_mean_ = adjusted_.mean(axis=0)
                adjusted_skew_ = stats.skew(adjusted_, axis=0, nan_policy="omit")
                adjusted_mode_ = stats.mode(adjusted_, axis=0, nan_policy="omit").mode[0]
                adjusted_cvar_ = stats.variation(adjusted_, axis=0, nan_policy="omit")
                adjusted_kurt_ = stats.kurtosis(adjusted_, axis=0, nan_policy="omit")
                adjusted_var_  = adjusted_.var(ddof=1, axis=0)
                adjusted_stddev_ = adjusted_.std(ddof=1, axis=0)
                adjusted_med_  = np.median(adjusted_, axis=0)
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


    def _bs_data_transform(self, which, q):
        """
        Starts with ``BaseChainLadderResult``'s ``_data_transform``, and
        performs additional pre-processing in order to generate plot of
        bootstrapped reserve ranges by origin period.

        Returns
        -------
        pd.DataFrame
        """
        data = self._data_transform()
        dfsims = self.get_quantile(q=q, which=which, symmetric=True)
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

        data.drop(["_ff", "_minf"], axis=1, inplace=True)
        dfv = data[["origin", "dev", "rectype", "value"]]
        dfl = data[["origin", "dev", "rectype", pctl_hdrs[0]]]
        dfu = data[["origin", "dev", "rectype", pctl_hdrs[-1]]]
        dfl["rectype"] = pctl_hdrs[0]
        dfl = dfl.rename({pctl_hdrs[0]:"value"}, axis=1)
        dfu["rectype"] = pctl_hdrs[-1]
        dfu = dfu.rename({pctl_hdrs[-1]:"value"}, axis=1)
        return(pd.concat([dfv, dfl, dfu]).sort_index())


    def plot(self, which="ultimate", q=.90, actuals_color="#334488",
             forecasts_color="#FFFFFF", fill_color="#FCFCB1", fill_alpha=.75,
             axes_style="darkgrid", context="notebook", col_wrap=5, hue_kws=None,
             **kwargs):
        """
        Generate exhibit representing the distribution of ultimates or
        reserves resulting from bootstrap resampling, along with
        percentiles from the distribution given by ``q``. Distribution by
        development period can be viewed as a function of origin period or
        in aggregate, controlled by ``view``.

        Parameters
        ----------
        which: {"ultimate", "reserve"}
            Specifies whether exhibit should reflect ultimate or reserve
            variability. Defaults to "ultimate".

        q: float in range of [0,1]
            Symmetric percentile interval to highlight, which must be between
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

        hue_kws: dictionary of param:list of values mapping
            Other keyword arguments to insert into the plotting call to let
            other plot attributes vary across levels of the hue variable
            (e.g. the markers in a scatterplot). Each list of values should
            have length 4, with each index representing aesthetic
            overrides for forecasts, actuals, lower percentile and upper
            percentile renderings respectively. Defaults to ``None``.

        kwargs: dict
            Additional styling options for scatter points. This can override
            default settings for ``plt.plot`` objects.

        Returns
        -------
        None
        """
        # which           = "ultimate"                               #
        # q               = .90                                      #
        # actuals_color   ="#334488"                                 #
        # forecasts_color ="#FFFFFF"                                 #
        # fill_color      = "#FCFCB1"                                #
        # fill_alpha      = .75                                      #
        # axes_style      = "darkgrid"                               #
        # context         = "notebook"                               #
        # col_wrap        = 5                                        #
        # hue_kws          = None                                    #
        # data            = bcl._bs_data_transform(which=which, q=q) #
        which_ = which.lower().strip()
        data = self._bs_data_transform(which=which_, q=q)
        pctl_hdrs = sorted([i for i in data["rectype"].unique() if i not in ("actual", "forecast")])
        sns.set_context(context)
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

            titlestr_ = "Bootstrap Chain Ladder {} Range Projections".format(which.title())

            grid_ = sns.FacetGrid(
                data, col="origin", hue="rectype", hue_kws=huekwargs,
                col_wrap=col_wrap, margin_titles=False, despine=True,
                sharex=True, sharey=True,
                hue_order=["forecast", "actual", pctl_hdrs[0], pctl_hdrs[-1]]
                )

            mean_ = grid_.map(plt.plot, "dev", "value",)
            grid_.set_axis_labels("", "")
            grid_.set(xticks=data["dev"].unique().tolist())
            grid_.set_titles("{col_name}", size=9)
            grid_.set_xticklabels(data["dev"].unique().tolist(), size=8)

            # Change ticklabel font size and place legend on each facet.
            for i, _ in enumerate(grid_.axes):
                ax_ = grid_.axes[i]
                legend_ = ax_.legend(
                    loc="upper left", fontsize="x-small", frameon=True,
                    fancybox=True, shadow=False, edgecolor="#909090",
                    framealpha=1, markerfirst=True,)
                legend_.get_frame().set_facecolor("#FFFFFF")
                ylabelss_ = [i.get_text() for i in list(ax_.get_yticklabels())]
                ylabelsn_ = [float(i.replace(u"\u2212", "-")) for i in ylabelss_]
                ylabelsn_ = [i for i in ylabelsn_ if i>=0]
                ylabels_ = ["{:,.0f}".format(i) for i in ylabelsn_]
                ax_.set_yticklabels(ylabels_, size=8)

                # Fill between upper and lower range bounds.
                axc_ = ax_.get_children()
                lines_ = [i for i in axc_ if isinstance(i, matplotlib.lines.Line2D)]
                xx = [i._x for i in lines_ if len(i._x)>0]
                yy = [i._y for i in lines_ if len(i._y)>0]
                x_, lb_, ub_ = xx[0], yy[-2], yy[-1]
                ax_.fill_between(x_, lb_, ub_, color=fill_color, alpha=fill_alpha)

                # Draw border around each facet.
                for _, spine_ in ax_.spines.items():
                    spine_.set_visible(True)
                    spine_.set_color("#000000")
                    spine_.set_linewidth(.50)

            # Adjust facets downward and and left-align figure title.
            plt.subplots_adjust(top=0.9)
            grid_.fig.suptitle(
                titlestr_, x=0.065, y=.975, fontsize=10, color="#404040", ha="left"
                )






    def hist(self, which="ultimate", color="#f33455", axes_style="darkgrid",
             context="notebook", col_wrap=5, **kwargs):
        """
        Generate visual representation of full predicitive distribtion
        of ultimates/reserves by origin and in aggregate. Additional
        options to style seaborn's distplot can be passed as keyword
        arguments.

        Parameters
        ----------
        which: {"ultimate", "reserve"}
            Specifies whether exhibit should reflect ultimate or reserve
            variability. Defaults to "ultimate".

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
        # which           = "ultimate"                               #
        # axes_style      = "darkgrid"                               #
        # context         = "notebook"                               #
        # col_wrap        = 5                                        #
        # kde             = True                                     #
        # rug             = True                                     #
        # norm            = True                                     #
        # which_          = which.lower().strip()                    #
        # kwargs          = {}                                       #
        # data              = bcl.sims_data[["sim", "origin", "dev", "rectype", "latest", "ultimate", "reserve",]] #
        which_ = which.lower().strip()
        data = self.sims_data[["sim", "origin", "dev", "rectype", "latest", "ultimate", "reserve",]]
        max_devp_ = data["dev"].max()
        data = data[(data["dev"]==max_devp_) & (data["rectype"]=="forecast")].reset_index(drop=True)

        sns.set_context(context)
        with sns.axes_style(axes_style):

            pltkwargs = {
                "color":"#f33455", "edgecolor":"#484848", "alpha":.75, "linewidth":.45,
                }

            titlestr_ = "Bootstrap Chain Ladder Predictive Distribution of {} by Origin".format(which.title())
            if kwargs is not None:
                pltkwargs.update(kwargs)

            grid_ = sns.FacetGrid(
                data, col="origin", col_wrap=col_wrap, margin_titles=False,
                despine=True, sharex=True, sharey=True,
                )
            hists_ = grid_.map(plt.hist, which_, **pltkwargs)


            grid_.set_axis_labels("", "")
            grid_.set(xticks=data["dev"].unique().tolist())
            grid_.set_titles("{col_name}", size=9)
            grid_.set_xticklabels("")

            # Change ticklabel font size and place legend on each facet.
            for i, _ in enumerate(grid_.axes):
                ax_ = grid_.axes[i]
                ylabelss_ = [i.get_text() for i in list(ax_.get_yticklabels())]
                ylabelsn_ = [float(i.replace(u"\u2212", "-")) for i in ylabelss_]
                ylabelsn_ = [i for i in ylabelsn_ if i>=0]
                ylabels_ = ["{:,.0f}".format(i) for i in ylabelsn_]
                ax_.set_yticklabels(ylabels_, size=8)
                ax_.grid(False)

                # Draw border around each facet.
                for _, spine_ in ax_.spines.items():
                    spine_.set_visible(True)
                    spine_.set_color("#000000")
                    spine_.set_linewidth(.50)

            # Adjust facets downward and and left-align figure title.
            plt.subplots_adjust(top=0.9)
            grid_.fig.suptitle(
                titlestr_, x=0.065, y=.975, fontsize=10, color="#404040", ha="left"
                )





    def plotdist(self, which="ultimate", view="aggregate", **kwargs):
        """
        Generate visual representation of full predicitive distribtion
        of loss reserves in aggregate or by origin. Additional options
        to style the histogram can be passed as keyword arguments.

        Parameters
        ----------
        view: str
            Set ``view`` to "origin" for faceted plot with predictive
            distirbution by origin year, or "aggregate" for single plot
            of predicitive distribution of reserves in aggregate. Default
            value is "aggregate".

        path: str
            If path is not None, save plot to the specified location.
            Otherwise, parameter is ignored. Default value is None.

        kwargs: dict
            Dictionary of optional matplotlib styling parameters.

        """
        plt_params = {"alpha":.995, "color":"#FFFFFF", "align":"mid",
                      "edgecolor":"black", "histtype":"bar",
                      "linewidth":1.1, "orientation":"vertical",}

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
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
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


    def get_quantile(self, q, which="reserve", symmetric=True, interpolation="linear"):
        """
        Return percentile of bootstrapped ultimate or reserve range
        distribution as specified by ``q``.

        Parameters
        ----------
        q: float in range of [0,1] (or sequence of floats)
            Percentile to compute, which must be between 0 and 1 inclusive.

        which: {"reserve", "ultimate"}
            The column used to compute bootstrapped confidence intervals.
            Default value is "reserve".

        symmetric: bool
            Whether the symmetric interval should be returned. For example, if
            ``symmetric==True`` and ``q=.95``, then the 2.5th and 97.5th
            quantiles of the bootstrapped reserve distribution will be returned
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
        which_ = which.lower().strip()
        if which_ not in self.sims_data.columns:
            raise ValueError("which must be one of ['reserve', 'ultimate']")

        dfsims = self.sims_data[["origin", "dev", which_]]
        pctl_ = np.asarray([q] if isinstance(q, (float, int)) else q)
        if np.any(np.logical_and(pctl_ <= 1, pctl_ >= 0)):
            if symmetric:
                pctlarr = np.sort(np.unique(np.append((1 - pctl_) / 2, (1 + pctl_) / 2)))
            else:
                pctlarr = np.sort(np.unique(pctl_))
        else:
            raise ValueError("Values for percentiles must fall between [0, 1].")

        pctlfmt = ["{:.5f}".format(i).rstrip("0").rstrip(".") + "%" for i in 100 * pctlarr]

        # Initialize DataFrame for percentile data.
        dfpctl = dfsims.groupby(["origin", "dev"]).aggregate(
            "quantile", q=.50, interpolation=interpolation)
        dfpctl = dfpctl.rename({which_:"50%"}, axis=1)
        dfpctl.columns.name = None

        for q_, pctlstr_ in zip(pctlarr, pctlfmt):
            if q_!=.50:
                df_ = dfsims.groupby(["origin", "dev"]).aggregate(
                    "quantile", q=q_, interpolation=interpolation)
                df_ = df_.rename({which_:pctlstr_}, axis=1)
                df_.columns.name = None
                dfpctl = dfpctl.join(df_)

        if .50 not in pctl_:
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
