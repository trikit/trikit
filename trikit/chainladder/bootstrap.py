"""
BootstrapChainLadder implementation.
"""
import functools
import warnings
import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy import stats
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
    1. England, P., and R. Verrall, (2002), *Stochastic Claims Reserving in General
       Insurance*, British Actuarial Journal 8(3): 443-518.

    2. CAS Working Party on Quantifying Variability in Reserve Estimates,
       *The Analysis and Estimation of Loss & ALAE Variability: A Summary Report*,
       Casualty Actuarial Society Forum, Fall 2005.

    3. Leong et al., (2012), *Back-Testing the ODP Bootstrap of the Paid
       Chain-Ladder Model with Actual Historical Claims Data*, Casualty Actuarial
       Society E-Forum.

    4. Kirschner, et al., *Two Approaches to Calculating Correlated Reserve
       Indications Across Multiple Lines of Business* Appendix III, Variance
       Journal, Volume 2/Issue 1.

    5. Shapland, Mark R., (2016), *Using the ODP Bootstrap Model: A
       Practicioner's Guide*, CAS Monograph Series Number 4: Casualty Actuarial
       Society, 2016.
    """
    def __init__(self, cumtri):
        """
        The BootstrapChainLadder class definition.

        Parameters
        ----------
        cumtri: triangle.CumTriangle
            A cumulative triangle instance.
        """
        super().__init__(cumtri=cumtri)

        self._dfrlvi = None
        self._dof = None




    def __call__(self, sims=1000, q=[.75, .95], neg_handler="all", procdist="gamma",
                 parametric=False, two_sided=False, interpolation="linear",
                 random_state=None):
        """
        ``BootstrapChainLadder`` simulation initializer. Generates predictive
        distribution of reserve outcomes by origin and in total.

        The estimated distribution of losses assumes development is complete
        by the final development period in order to avoid the complication of
        modeling a tail factor.

        Parameters
        ----------
        sims: int
            The number of bootstrap simulations to perform. Defaults to 1000.

        q: array_like of float or float
            Quantile or sequence of quantiles to compute, which must be
            between 0 and 1 inclusive.

        neg_handler: str
            If ``neg_handler="first"``, any first development period negative
            cells will be coerced to +1. If ``neg_handler="all"``, the minimum
            value in all triangle cells is identified (identified as 'MIN_CELL').
            If MIN_CELL is less than or equal to 0, (MIN_CELL + X = +1.0) is
            solved for X. X is then added to every other cell in the triangle,
            resulting in all triangle cells having a value strictly greater
            than 0.

        procdist: str
            The distribution used to incorporate process variance. Currently,
            this can only be set to "gamma".

        two_sided: bool
            Whether the two_sided interval should be included in summary
            output. For example, if ``two_sided==True`` and ``q=.95``, then
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
        BootstrapChainLadderResult
        """
        ldfs = self._ldfs(sel="all-weighted")
        cldfs = self._cldfs(ldfs=ldfs)
        ultimates = self._ultimates(cldfs=cldfs)
        reserves = self._reserves(ultimates=ultimates)
        maturity = self.tri.maturity.astype(np.str)
        latest = self.tri.latest_by_origin
        trisqrd = self._trisqrd(ldfs=ldfs)

        # Obtain reference to Bootstrap estimates.
        tri_fit_cum = self._tri_fit_cum(ldfs=ldfs)
        tri_fit_incr = self._tri_fit_incr(fitted_tri_cum=tri_fit_cum)
        unscld_residuals = self._resid_us(fitted_tri_incr=tri_fit_incr)
        adjust_residuals = self._resid_adj(resid_us=unscld_residuals)
        scale_param = self._scale_param(resid_us=unscld_residuals)
        sampling_dist = self._sampling_dist(resid_adj=adjust_residuals)
        dfsamples = self._bs_samples(
            sampling_dist=sampling_dist, fitted_tri_incr=tri_fit_incr,
            sims=sims, neg_handler=neg_handler, parametric=parametric,
            random_state=random_state
            )
        dfldfs = self._bs_ldfs(dfsamples=dfsamples)
        dfcombined = self._get_dfcombined(dfsamples, dfldfs)
        dfforecasts = self._bs_forecasts(dfsamples, dfldfs, scale_param)
        dfprocerror = self._bs_process_error(
            dfforecasts=dfforecasts, scale_param=scale_param, procdist=procdist,
            random_state=random_state
            )

        dfreserves = self._bs_reserves(dfprocerror=dfprocerror)
        qtls, qtlhdrs = self._qtls_formatter(q=q, two_sided=two_sided)

        # Compile Chain Ladder point estimate summary.
        dfmatur = maturity.to_frame().reset_index(drop=False).rename({"index":"origin"}, axis=1)
        dfcldfs = cldfs.to_frame().reset_index(drop=False).rename({"index":"maturity"}, axis=1)
        dfcldfs["maturity"] = dfcldfs["maturity"].astype(np.str)
        dfcldfs["emergence"] = 1 / dfcldfs["cldf"]
        dfsumm = dfmatur.merge(dfcldfs, on=["maturity"], how="left").set_index("origin")
        dfsumm.index.name = None
        dflatest = latest.to_frame().rename({"latest_by_origin":"latest"}, axis=1)
        dfsumm = functools.reduce(
            lambda df1, df2: df1.join(df2),
            (dflatest, ultimates.to_frame(), reserves.to_frame()),
            dfsumm
            )

        # Attach quantiles.
        for ii, jj in zip(qtls, qtlhdrs):
            dfsumm[jj] = dfsumm.index.map(
                lambda v: np.percentile(
                    dfreserves[dfreserves.origin==v]["reserve"].values,
                    100 * ii, interpolation=interpolation
                    )
                )

        # Add "Total" index and set to NaN fields that shouldn't be aggregated.
        dfsumm.loc["total"] = dfsumm.sum()
        dfsumm.loc["total", "maturity"] = ""
        dfsumm.loc["total", ["cldf", "emergence"]] = np.NaN

        kwds = {
            "sel":"all-weighted", "sims":sims, "neg_handler":neg_handler,
            "procdist":procdist, "parametric":parametric, "q":q,
            "interpolation":interpolation,
            }

        # Instantiate and return BootstrapChainLadderResult instance.
        bcl_result = BootstrapChainLadderResult(
            summary=dfsumm, tri=self.tri, ldfs=ldfs, tail=1.0, trisqrd=trisqrd,
            reserve_dist=dfreserves, sims_data=dfprocerror, scale_param=scale_param,
            dof=self.dof, unscaled_residuals=unscld_residuals,
            adjusted_residuals=adjust_residuals,
            sampling_dist=None if parametric else sampling_dist,
            fitted_tri_cum=tri_fit_cum, fitted_tri_incr=tri_fit_incr, **kwds
            )

        return(bcl_result)


    @staticmethod
    def _qtls_formatter(q, two_sided=False):
        """
        Return array_like of formatted quantiles for MackChainLadder
        summary.

        Parameters
        ----------
        q: array_like of float
            Sequence of quantiles to compute, which must be
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
    def dfrlvi(self):
        """
        Transform triangle's last valid origin index into DataFrame format.

        Returns
        -------
        pd.DataFrame
        """
        if self._dfrlvi is None:
            df = self.tri.rlvi.reset_index(drop=False)
            df = df.rename({"index":"origin", "dev":"l_act_dev"}, axis=1)
            self._dfrlvi = df.drop("col_offset", axis=1)
        return(self._dfrlvi)


    def _get_dfcombined(self, dfsamples, dfldfs):
        """
        Merge output of ``self._bs_samples`` and ``self._bs_ldfs``.

        Parameters
        ----------
        dfsamples: pd.DataFrame
            Output from ``self._bs_samples``.

        dfldfs: pd.DataFrame
            Output from ``self._bs_ldfs``.

        Returns
        -------
        pd.DataFrame
        """
        dfcombined = dfsamples.merge(dfldfs, on=["sim", "dev"], how="left")
        dfcombined = dfcombined.merge(self.dfrlvi, on=["origin"], how="left")
        return(dfcombined.reset_index(drop=True).sort_values(by=["sim", "origin", "dev"]))


    @property
    def dof(self):
        """
        Return the degress of freedom.

        Returns
        -------
        int
        """
        if self._dof is None:
            self._dof = self.tri.nbr_cells - (self.tri.columns.size - 1) + self.tri.index.size
        return(self._dof)


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
        return((resid_us**2).sum().sum() / self.dof)


    def _tri_fit_cum(self, ldfs):
        """
        Return the cumulative fitted triangle using backwards recursion,
        starting with the observed cumulative paid/incurred-to-date along the
        latest diagonal.

        Parameters
        ----------
        ldfs: pd.Series
            Selected ldfs, typically the output of calling ``self._ldfs``.

        Returns
        -------
        pd.DataFrame
        """
        fitted_tri_cum = self.tri.copy(deep=True)

        for ii in range(fitted_tri_cum.shape[0]):
            iterrow = fitted_tri_cum.iloc[ii, :]
            if iterrow.isnull().any():
                # Find first NaN element in iterrow.
                nan_hdr = iterrow.isnull()[iterrow.isnull()==True].index[0]
                nan_idx = fitted_tri_cum.columns.tolist().index(nan_hdr)
                init_idx = nan_idx - 1
            else:
                # If here, iterrow is the most mature exposure period.
                init_idx = fitted_tri_cum.shape[1] - 1

            # Set to NaN any development periods earlier than init_idx.
            fitted_tri_cum.iloc[ii, :init_idx] = np.NaN

            # Iterate over rows, undeveloping triangle from latest diagonal.
            for j in range(fitted_tri_cum.iloc[ii, :init_idx].size, 0, -1):
                prev_col_idx, curr_col_idx, curr_ldf_idx = j, j - 1, j - 1
                prev_col_val = fitted_tri_cum.iloc[ii, prev_col_idx]
                curr_ldf_val = ldfs.iloc[curr_ldf_idx]
                fitted_tri_cum.iloc[ii, curr_col_idx] = (prev_col_val / curr_ldf_val)
        return(fitted_tri_cum)


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
        tri = fitted_tri_cum.diff(axis=1)
        tri.iloc[:, 0] = fitted_tri_cum.iloc[:, 0]
        return(tri)


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
        # I represents actual incremental losses, m fitted incremental losses.
        I = pd.DataFrame(self.tri.to_incr())
        m = fitted_tri_incr
        return((I - m) / np.sqrt(m.abs()))


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
        return(np.sqrt(self.tri.nbr_cells / self.dof) * resid_us)


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
                    neg_handler="all", parametric=False, random_state=None):
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

        neg_handler: str
            If ``neg_handler="first"``, any first development period negative
            cells will be coerced to +1. If ``neg_handler="all"``, the minimum
            value in all triangle cells is identified (identified as 'MIN_CELL').
            If MIN_CELL is less than or equal to 0, (MIN_CELL + X = +1.0) is
            solved for X. X is then added to every other cell in the triangle,
            resulting in all triangle cells having a value strictly greater
            than 0.

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

        sampling_dist = sampling_dist.flatten()

        fti = fitted_tri_incr.reset_index(drop=False).rename({"index":"origin"}, axis=1)
        dfm = pd.melt(fti, id_vars=["origin"], var_name="dev", value_name="value")
        dfm = dfm[~np.isnan(dfm["value"])].astype({"origin":np.int, "dev":np.int, "value":np.float})

        # Make positive any first development period negative values.
        if np.any(dfm["value"]<=0):

            if neg_handler=="first":
                min_devp = dfm["dev"].min()
                dfm["value"] = np.where(
                    np.logical_and(dfm["dev"].values==min_devp, dfm["value"].values<0),
                    1., dfm["value"].values
                    )

            elif neg_handler=="all":
                # Add the absolute value of the minimum triangle amount plus
                # one to all other triangle  cells.
                dfm["value"] = dfm["value"] + np.abs(dfm["value"].min()) + 1

            else:
                raise ValueError("`neg_handler` must be in ['first', 'all'].")


        dfi = self.tri.to_tbl(drop_nas=False).drop("value", axis=1)
        dfp = dfi.merge(dfm, how="outer", on=["origin", "dev"])
        dfp["rectype"] = np.where(np.isnan(dfp["value"].values), "forecast", "actual")
        dfp = dfp.rename({"value":"incr"}, axis=1)
        dfp["incr_sqrt"] = np.sqrt(dfp["incr"].values)
        dfrtypes = {"origin":np.int, "dev":np.int, "incr":np.float,
                   "incr_sqrt":np.float, "rectype":np.str,}
        dfrcols = ["origin", "dev", "incr", "rectype", "incr_sqrt"]

        # Replicate dfp sims times then redefine datatypes.
        dfr = pd.DataFrame(np.tile(dfp, (sims, 1)), columns=dfrcols).astype(dfrtypes)

        # Assign simulation identifier to each record in dfr.
        dfr["sim"] = np.divmod(dfr.index, self.tri.shape[0] * self.tri.shape[1])[0]
        sample_size = dfr.shape[0]

        if parametric:
            # Sample random standard normal residuals.
            dfr["resid"] = prng.normal(loc=0, scale=sampling_dist.std(ddof=1), size=sample_size)
        else:
            # Randomly sample residuals from sampling_dist.
            dfr["resid"] = prng.choice(sampling_dist, sample_size, replace=True)

        # Calcuate resampled incremental and cumulative losses.
        dfr["resid"] = np.where(dfr["rectype"].values=="forecast", np.NaN, dfr["resid"].values)
        dfr = dfr.sort_values(by=["sim", "origin", "dev"]).reset_index(drop=True)
        dfr["samp_incr"] = dfr["incr"].values + dfr["resid"].values * dfr["incr_sqrt"].values
        dfr["samp_cum"]  = dfr.groupby(["sim", "origin"], as_index=False)["samp_incr"].cumsum()
        return(dfr.reset_index(drop=True))


    def _bs_ldfs(self, dfsamples):
        """
        Compute and return loss development factors for each set of synthetic
        loss data.

        Parameters
        ----------
        dfsamples: pd.DataFrame
            Output from ``self._bs_samples``.

        Returns
        -------
        pd.DataFrame
        """
        keepcols = ["sim", "origin", "dev", "samp_cum", "last_origin"]
        new_col_names = {"index":"dev", "origin":"last_origin", "row_offset":"origin_offset"}
        dflvi = self.tri.clvi.reset_index(drop=False).rename(new_col_names, axis=1)
        dfinit = dfsamples.merge(dflvi, how="left", on=["dev"])
        dfinit = dfinit[keepcols].sort_values(by=["sim", "dev", "origin"])
        df = dfinit[~np.isnan(dfinit["samp_cum"])].reset_index(drop=True)
        df["_aggdev2"] = np.where(df["origin"].values==df["last_origin"].values, 0, df["samp_cum"].values)
        df2 = df.groupby(["sim", "dev"], as_index=False)[["samp_cum", "_aggdev2"]].sum().rename(
            {"samp_cum":"_aggdev1"}, axis=1)
        df2["_aggdev2"] = df2["_aggdev2"].shift(periods=1)
        df2["dev"] = df2["dev"].shift(periods=1)
        dfldfs = df2[df2["_aggdev2"]!=0].dropna(how="any")
        dfldfs["dev"] = dfldfs["dev"].astype(np.int)
        dfldfs["ldf"] = dfldfs["_aggdev1"] / dfldfs["_aggdev2"]
        return(dfldfs[["sim", "dev", "ldf"]].reset_index(drop=True))



    def _bs_forecasts(self, dfsamples, dfldfs, scale_param):
        """
        Populate lower-right of each simulated triangle using values from
        ``self._bs_samples`` and development factors from ``self._bs_ldfs``.

        Parameters
        ----------
        Parameters
        ----------
        dfsamples: pd.DataFrame
            Output from ``self._bs_samples``.

        dfldfs: pd.DataFrame
            Output from ``self._bs_ldfs``.

        scale_param: float
            the sum of the squared unscaled Pearson residuals over the
            degrees of freedom. Output from ``self._scale_param``.

        Returns
        -------
        pd.DataFrame
        """
        dfcombined = self._get_dfcombined(dfsamples, dfldfs)
        min_origin_year = dfcombined["origin"].min()
        dfcombined["_l_init_indx"] = np.where(
            dfcombined["dev"].values>=dfcombined["l_act_dev"].values, dfcombined.index.values, -1)
        dfacts = dfcombined[(dfcombined["origin"].values==min_origin_year) | (dfcombined["_l_init_indx"].values==-1)]
        dffcst = dfcombined[~dfcombined.index.isin(dfacts.index)].sort_values(by=["sim", "origin", "dev"])
        dffcst["_l_act_indx"] = dffcst.groupby(["sim", "origin"])["_l_init_indx"].transform("min")
        dffcst["l_act_cum"] = dffcst.lookup(dffcst["_l_act_indx"].values, ["samp_cum"] * dffcst.shape[0])
        dffcst["_cum_ldf"] = dffcst.groupby(["sim", "origin"])["ldf"].transform("cumprod").shift(periods=1)
        dffcst["_samp_cum2"] = dffcst["l_act_cum"].values * dffcst["_cum_ldf"].values
        dffcst["_samp_cum2"] = np.where(np.isnan(dffcst["_samp_cum2"].values), 0, dffcst["_samp_cum2"].values)
        dffcst["cum_final"] = np.where(np.isnan(dffcst["samp_cum"].values), 0, dffcst["samp_cum"].values) + dffcst["_samp_cum2"].values

        # Combine forecasts with actuals then compute incremental losses by sim and origin.
        dffcst = dffcst.drop(labels=["samp_cum", "samp_incr"], axis=1).rename(columns={"cum_final":"samp_cum"})
        dfsqrd = pd.concat([dffcst, dfacts], sort=True).sort_values(by=["sim", "origin", "dev"])
        dfsqrd["_dev1_ind"] = (dfsqrd["dev"].values==1) * 1
        dfsqrd["_incr_dev1"] = dfsqrd["_dev1_ind"].values * dfsqrd["samp_cum"].values
        dfsqrd["_incr_dev2"] = dfsqrd.groupby(["sim", "origin"])["samp_cum"].diff(periods=1)
        dfsqrd["_incr_dev2"] = np.where(np.isnan(dfsqrd["_incr_dev2"].values), 0, dfsqrd["_incr_dev2"].values)
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
        ``scale_param``.

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
            will result in an error.

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
            dfforecasts["samp_incr"].values
            )
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
        keepcols = ["sim", "origin", "latest", "ultimate", "reserve"]
        max_devp = dfprocerror["dev"].values.max()
        dfprocerror["reserve"] = dfprocerror["ultimate"] - dfprocerror["latest"]
        dfreserves = dfprocerror[dfprocerror["dev"].values==max_devp][keepcols].drop_duplicates()
        dfreserves["latest"]  = np.where(
            np.isnan(dfreserves["latest"].values),
            dfreserves["ultimate"].values, dfreserves["latest"].values
            )
        dfreserves["reserve"] = np.nan_to_num(dfreserves["reserve"].values, 0)
        return(dfreserves.sort_values(by=["origin", "sim"]).reset_index(drop=True))





class BootstrapChainLadderResult(BaseChainLadderResult):
    """
    BootstrapChainLadder output.
    """
    def __init__(self, summary, tri, ldfs, tail, trisqrd, reserve_dist, sims_data,
                 scale_param, dof, unscaled_residuals, adjusted_residuals,
                 sampling_dist, fitted_tri_cum, fitted_tri_incr, **kwargs):
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

        tri: trikit.triangle.CumTriangle
            A cumulative triangle instance.

        ldfs: pd.Series
            Loss development factors.

        scale_param: float
            The the sum of the squared unscaled Pearson residuals over the
            triangle's degrees of freedom.

        dof: int
            Triangle degrees of freedom.

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
            Additional keyword arguments passed into ``BootstrapChainLadder``'s
            ``__call__`` method.
        """
        super().__init__(summary=summary, tri=tri, ldfs=ldfs, tail=tail,
                         trisqrd=trisqrd, **kwargs)

        self.unscaled_residuals = unscaled_residuals
        self.adjusted_residuals = adjusted_residuals
        self.fitted_tri_incr = fitted_tri_incr
        self.fitted_tri_cum = fitted_tri_cum
        self.sampling_dist = sampling_dist
        self.reserve_dist = reserve_dist
        self.scale_param = scale_param
        self.sims_data = sims_data
        self.summary = summary
        self.trisqrd = trisqrd
        self.ldfs = ldfs
        self.tail = tail
        self.dof = dof
        self.tri = tri

        # Create DataFrame with reserve and specified quantile estimates.
        qtlsfields = [i for i in self.summary.columns if i.endswith("%")]
        self.dfqtls = self.summary[["reserve"] + qtlsfields]

        if kwargs is not None:
            for kk in kwargs:
                setattr(self, kk, kwargs[kk])

        self.qtlhdrs = {i:"{:,.0f}".format for i in qtlsfields}
        self.qtlhdrs.update({"std_error":"{:,.0f}".format, "cv":"{:.5f}".format})
        self._summspecs.update(self.qtlhdrs)

        # Properties.
        self._aggregate_distribution = None
        self._origin_distribution = None
        self._residuals_detail = None
        self._fit_assessment = None


    def _qtls_formatter(self, q, two_sided=False):
        """
        Return array_like of actual and formatted quantiles for BootstrapChainLadder
        summary.

        Parameters
        ----------
        q: array_like of float or float
            Quantile or sequence of quantiles to compute, which must be
            between 0 and 1 inclusive.

        Returns
        -------
        tuple of ndarray
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
        bootstrap resampling. Applicable when ``parametric=True``. Returns a
        dictionary with keys ``kstest``, ``anderson``, ``shapiro``, ``skewtest``,
        ``kurtosistest`` and ``normaltest``, corresponding to statistical
        tests available in ``scipy.stats``.

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




    def _bs_data_transform(self, qtls, qtlhdrs):
        """
        Starts with ``BaseChainLadderResult``'s ``_data_transform``, and
        performs additional pre-processing in order to generate plot of
        bootstrapped reserve ranges by origin period.

        Returns
        -------
        pd.DataFrame
        """
        data0 = self._data_transform()
        data0 = data0[data0["origin"]!="total"]
        data1 = self._get_quantiles_by_devp(qtls, qtlhdrs)
        data1 = data1[data1["origin"]!="total"]
        data  = data0.merge(data1, on=["origin", "dev"], how="left")

        # Remove qtlhdrs values where rectype=="actual".
        for qtlhdr in qtlhdrs:
            data[qtlhdr] = np.where(
                data["rectype"].values=="actual", np.NaN, data[qtlhdr].values
                )

        # Determine the first forecast period by origin, and set q-fields to actuals.
        increment = np.unique(self.ldfs.index[1:] - self.ldfs.index[:-1])[0]
        data["_ff"] = np.where(
            data["rectype"].values=="forecast",
            data["dev"].values, data["dev"].values.max() + increment
            )
        data["_minf"] = data.groupby(["origin"])["_ff"].transform("min")
        for hdr in qtlhdrs:
            data[hdr] = np.where(
                np.logical_and(
                    data["rectype"].values=="forecast",
                    data["_minf"].values==data["dev"].values
                    ), data["loss"].values, data[hdr].values
                )

        data = data.drop(["_ff", "_minf"], axis=1).reset_index(drop=True)
        dfv = data[["origin", "dev", "rectype", "loss"]]
        dfl = data[["origin", "dev", "rectype", qtlhdrs[0]]]
        dfu = data[["origin", "dev", "rectype", qtlhdrs[-1]]]
        dfl["rectype"] = qtlhdrs[0]
        dfl = dfl.rename({qtlhdrs[0]:"loss"}, axis=1)
        dfu["rectype"] = qtlhdrs[-1]
        dfu = dfu.rename({qtlhdrs[-1]:"loss"}, axis=1)
        return(pd.concat([dfv, dfl, dfu]).sort_index().reset_index(drop=True))


    def _get_quantiles_by_devp(self, qtls, qtlhdrs):
        """
        Get quantile of boostrapped reserve distribution for an individual
        origin period and in total.

        Parameters
        ----------
        q: array_like
            A length-2 sequence representing to upper and lower bounds of
            the estimated reserve distribution.

        Returns
        -------
       pd.DataFrame
        """
        # qtls, qtlhdrs = self._qtls_formatter(q=q, two_sided=False)
        dfsims = self.sims_data[["origin", "dev", "ultimate",]]
        dfults = dfsims[dfsims.dev==dfsims.dev.max()].reset_index(drop=True)
        dev_increment = np.unique(self.ldfs.index[1:] - self.ldfs.index[:-1])[0]
        dfults["dev"] = self.ldfs.index.max() + dev_increment
        dfsims = pd.concat([dfsims, dfults])
        dftotal_keys = dfsims[dfsims.origin==dfsims.origin.min()][["origin", "dev"]].drop_duplicates()
        dftotal_keys["origin"] = "total"
        dfqtls_keys = pd.concat(
            [dfsims[["origin", "dev"]].drop_duplicates(), dftotal_keys]
            ).reset_index(drop=True
            )

        # Get total reserve across all origin periods.
        dftotal = dfsims.copy(deep=True)
        dftotal["origin"] = "total"
        dftotal = dftotal.groupby(["origin", "dev"], as_index=False)

        dflist = []
        for ii,jj in zip(qtls, qtlhdrs):
            dfqtl = dfsims.groupby(["origin", "dev"], as_index=False).aggregate(
                "quantile", q=ii, interpolation="linear").rename(
                {"ultimate":jj}, axis=1
                )
            dftotal_qtl = dftotal.aggregate(
                "quantile", q=ii, interpolation="linear").rename({"ultimate":jj},
                axis=1
                )
            dflist.append(pd.concat([dfqtl, dftotal_qtl]))

        # Combine DataFrames in dflist into single table.
        dfqtls = functools.reduce(
            lambda df1, df2: df1.merge(df2, on=["origin", "dev"], how="left"),
            dflist, dfqtls_keys).reset_index(drop=True)
        return(dfqtls)


    def plot(self, q=.90, actuals_color="#334488", forecasts_color="#FFFFFF",
             fill_color="#FCFCB1", fill_alpha=.75, axes_style="darkgrid",
             context="notebook", col_wrap=4, hue_kws=None, exhibit_path=None,
             **kwargs):
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
            Color or hexidecimal color code used to represent actuals.
            Defaults to "#00264C".

        forecasts_color: str
            Color or hexidecimal color code used to represent forecasts.
            Defaults to "#FFFFFF".

        fill_color: str
            Color or hexidecimal color code used to represent the fill color
            between reserve distribution quantiles associated with ``q``.
            Defaults to "#FCFCB1".

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

        exhibit_path: str
            Path to which exhibit should be written. If None, exhibit will be
            rendered via ``plt.show()``.

        kwargs: dict
            Additional styling options for scatter points. This should include
            additional options accepted by ``plt.plot``.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_context(context)

        qtls, qtlhdrs = self._qtls_formatter(q=q, two_sided=True)
        data = self._bs_data_transform(qtls, qtlhdrs).dropna(how="any", subset=["loss"])

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

            grid = sns.FacetGrid(
                data, col="origin", hue="rectype", hue_kws=huekwargs,
                col_wrap=col_wrap, margin_titles=False, despine=True,
                sharex=False, sharey=False,
                hue_order=["forecast", "actual", qtlhdrs[0], qtlhdrs[-1]]
                )

            ult_vals = grid.map(plt.plot, "dev", "loss",)
            devp_xticks = np.sort(data.dev.unique())
            devp_xticks_str = [
                str(ii) if ii!=devp_xticks.max() else "ult" for ii in devp_xticks
                ]
            grid.set(xticks=devp_xticks);  grid.set_xticklabels(devp_xticks_str, size=7)

            with warnings.catch_warnings():

                warnings.simplefilter("ignore")

                # Change ticklabel font size and place legend on each facet.
                for origin, ax_ii in zip(np.sort(data.origin.unique()), grid.axes):

                    legend = ax_ii.legend(
                        loc="lower right", fontsize="small", frameon=True,
                        fancybox=True, shadow=False, edgecolor="#909090",
                        framealpha=1, markerfirst=True,)
                    legend.get_frame().set_facecolor("#FFFFFF")

                    # For given origin, determine optimal 5-point tick labels.
                    origin_max_val = data[data.origin==origin].loss.max()
                    y_ticks, y_ticklabels = self._get_yticks(origin_max_val)
                    ax_ii.set_yticks(y_ticks)
                    ax_ii.set_yticklabels(y_ticklabels, size=7)



                    ax_ii.annotate(
                        origin, xy=(.075, .90), xytext=(.075, .90), xycoords='axes fraction',
                        textcoords='axes fraction', fontsize=9, rotation=0, color="#000000",
                        )
                    ax_ii.set_title(""); ax_ii.set_xlabel(""); ax_ii.set_ylabel("")

                    # Fill between upper and lower range bounds.
                    axc = ax_ii.get_children()
                    lines_ = [jj for jj in axc if isinstance(jj, matplotlib.lines.Line2D)]
                    xx = [jj._x for jj in lines_ if len(jj._x)>0]
                    yy = [jj._y for jj in lines_ if len(jj._y)>0]
                    x_, lb_, ub_ = xx[0], yy[-2], yy[-1]
                    ax_ii.fill_between(x_, lb_, ub_, color=fill_color, alpha=fill_alpha)

                    # Draw border around each facet.
                    for _, spine_ in ax_ii.spines.items():
                        spine_.set(visible=True, color="#000000", linewidth=.50)

            if exhibit_path is not None:
                plt.savefig(exhibit_path)
            else:
                plt.show()


    def hist(self, color="#FFFFFF", axes_style="darkgrid", context="notebook",
             col_wrap=4, exhibit_path=None, **kwargs):
        """
        Generate histogram of estimated reserve distribution by accident
        year and in total.

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

        exhibit_path: str
            Path to which exhibit should be written. If None, exhibit will be
            rendered via ``plt.show()``.

        kwargs: dict
            Dictionary of optional matplotlib styling parameters.

        """
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_context(context)

        data0 = self.sims_data[["sim", "origin", "dev", "rectype", "latest", "reserve",]]
        data0 = data0[(data0["dev"]==data0["dev"].max()) & (data0["rectype"]=="forecast")].reset_index(drop=True)
        data0 = data0.drop(["dev", "rectype", "latest"], axis=1)

        # Include additional origin representing aggregate distribution.
        data1 = data0.groupby("sim", as_index=False)[["reserve"]].sum()
        data1["origin"] ="total"
        data = pd.concat([data0, data1])

        # Get mean, min and max ultimate and reserve by origin.
        med_data = data.groupby("origin", as_index=False)[["reserve"]].median().rename(
            {"reserve":"med_res"}, axis=1).set_index("origin")
        min_data = data.groupby("origin", as_index=False)[["reserve"]].min().rename(
            {"reserve":"min_res"}, axis=1).set_index("origin")
        max_data = data.groupby("origin", as_index=False)[["reserve"]].max().rename(
            {"reserve":"max_res"}, axis=1).set_index("origin")
        dfmetrics = functools.reduce(lambda df1, df2: df1.join(df2), (med_data, min_data, max_data))
        dfmetrics = dfmetrics.applymap(lambda v: 0 if v<0 else v).reset_index(drop=False)

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

            hists = grid.map(plt.hist, "reserve", **pltkwargs)
            grid.set_axis_labels("", "")
            grid.set_titles("", size=6)

            # Change ticklabel font size and place legend on each facet.
            origin_vals = sorted([int(ii) for ii in data["origin"].unique() if ii!="total"])
            dindex = {jj:ii for ii,jj in enumerate(origin_vals)}
            dindex.update({"total":max(dindex.values())+1})
            data["origin_index"] = data["origin"].map(dindex)
            origin_order = data[["origin_index", "origin"]].drop_duplicates().sort_values(
                "origin_index"
                ).origin.values

            with warnings.catch_warnings():

                warnings.simplefilter("ignore")

                for origin, ax_ii in zip(origin_order, grid.axes):

                    xmin = np.max([0, dfmetrics[dfmetrics.origin==origin]["min_res"].item()])
                    xmax = dfmetrics[dfmetrics.origin==origin]["max_res"].item() * 1.025
                    xmed = dfmetrics[dfmetrics.origin==origin]["med_res"].item()
                    origin_str = "{} reserve".format(origin)
                    ax_ii.set_xlim([0, xmax])
                    ax_ii.axvline(xmed, color="#E02C70", linestyle="--", linewidth=1.5)
                    ax_ii.grid(False)

                    ymedloc = max(rect.get_height() for rect in ax_ii.patches) * .30
                    ax_ii.set_yticks([]); ax_ii.set_yticklabels([])
                    ax_ii.tick_params(
                        axis="x", which="both", bottom=True, top=False, labelbottom=True
                        )
                    ax_ii.set_xticklabels(
                        ["{:,.0f}".format(jj) for jj in ax_ii.get_xticks()], size=7
                        )
                    ax_ii.annotate(
                        origin_str, xy=(.65, .925), xycoords='axes fraction',
                        textcoords='axes fraction', fontsize=8, rotation=0, color="#000000",
                        )
                    ax_ii.annotate(
                        "median = {:,.0f}".format(xmed), (xmed, ymedloc), xytext=(7.5, 0),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7,
                        rotation=90, color="#000000"
                        )

                    # Draw border around each facet.
                    for _, spine in ax_ii.spines.items():
                        spine.set(visible=True, color="#000000", linewidth=.50)

            if exhibit_path is not None:
                plt.savefig(exhibit_path)
            else:
                plt.show()


