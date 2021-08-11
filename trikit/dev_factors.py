"""
Loss development factor class using the approach outlined in Barnett and
Zehnwirth[1] in which:

    - delta=0:
        Weighted average of link ratios weighted by volume squared.

    - delta=1:
        Weighted link ratio average by volume (Chain Ladder average method),

    - delta=2:
        Simple arithmetic average of link ratios.

As stated in [1], one of the advantages of estimating link-ratios using
regressions is that both standard errors of the average method selection and
standard errors of the forecasts can be obtained. Another more important
advantage is that the assumptions made by the method can be tested.


References
----------
1. Mack, Thomas, (1999), *The Standard Error of Chain Ladder Reserve Estimates:
   Recursive Calculation and Inclusion of a Tail Factor*, ASTIN Bulletin 29,
   no. 2:361-366.

2. G. Barnett and B. Zehnwirth, (2000),  *Best Estimates for Reserves*,
   Proceedings of the CAS Volume LXXXVII, Numbers 166 & 167.


Basic (unweighted) linear regression through the origin: lm(y~x + 0)

Basic weighted linear regression through the origin: lm(y~x + 0, weights=weights)

Volume weighted chain-ladder age-to-age factors: lm(y~x + 0, weights=1/x)

Simple average of age-to-age factors: lm(y~x + 0, weights=1/x^2)

Barnett & Zehnwirth (2000) use delta = 0, 1, 2 to distinguish between the above
three different regression approaches: lm(y~x + 0, weights=weights/x^delta).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf



class _DevelopmentFactors:

    def __init__(self, cumtri, data=None, tail=1):
        """
        Loss development factors class.

        Parameters
        ----------
        cumtri: triangle.CumTriangle
            Cumulative triangle instance.

        data: array_like
            Custom set of loss development factors. If None, ldfs will be computed
            based on ``cumtri``.

        delta: {0, 1, 2}
            The type of average to compute:

            - delta=0:
                Weighted average of link ratios weighted by volume squared.
            - delta=1:
                Weighted link ratio average by volume (Chain Ladder average method).
            - delta=2:
                Simple arithmetic average of link ratios.

            Default value is 1.

        tail: float
            Tail factor. Defaults to 1.0.

        data: triangle.CumTriangle or array_like
        """
        self.tri = cumtri
        self.data = data
        self.tail = tail








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
        if weights is None:
            w = np.ones(len(vals))
        else:
            w = weights
            if len(w)!=len(vals):
                raise ValueError("`vals` and `weights` must have same size")
        arr = np.asarray(vals, dtype=np.float)
        return((w * arr).sum() / w.sum())




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
        if weights is None:
            w = np.ones(len(vals))
        else:
            w = weights
            if len(w)!=len(vals):
                raise ValueError("`vals` and `weights` must have same size")

        # Return first element of arr_all if all array elements are the same.
        arr_all = np.sort(np.asarray(vals, dtype=np.float))
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


    def _ldfs_lm(self, delta=1, nbr_periods=None):
        """
        Regress development period n+1 on development period n.

        Parameters
        ----------
        delta: {0, 1, 2}
            The type of average to compute:

            - delta=0:
                Weighted average of link ratios weighted by volume squared.
            - delta=1:
                Weighted link ratio average by volume (Chain Ladder average method).
            - delta=2:
                Simple arithmetic average of link ratios.

            Default value is 1.

        nbr_periods: int
            The number of periods to retain when computing an ``n`` period average
            development factor. If None, retain all available periods. Default
            value is None.

        Returns
        -------
        pd.Series


        lm(y~x + 0, weights=weights/x^delta).
        """
        delta = 1
        # Remove triangle values with associated weight of 0.
        data = (tri.weights * tri).replace(0, np.NaN)

        # Create empty Series object to hold ldf estimates.
        ldfs = pd.Series(index=tri.devp[:-1].values, dtype=np.float, name="ldf")

        devp0, devp1 = tri.devp.values[:-1], tri.devp.values[1:]
        dresults_keys = ["{}-{}".format(ii,jj) for ii,jj in zip(devp0, devp1)]
        dresults = {jj:{} for jj in dresults_keys}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

        for indx_x, devp_x in enumerate(data.columns[:-1]):

            # indx_x = 0
            # devp_x = 1

            indx_y = indx_x + 1
            devp_y = tri.devp.values[indx_y]
            dfxy = data.loc[:,devp_x:devp_y].rename({devp_x:"x", devp_y:"y"}, axis=1).dropna(how="any")
            mdl = smf.wls(
                formula="y ~ x - 1", data=dfxy, weights=1/dfxy["x"].values**delta
                ).fit()

            mdl_key = "{}-{}".format(devp_x, devp_y)
            ldfs[devp_x] = mdl.params.item()
            dresults[mdl_key]["model"] = mdl
            dresults[mdl_key]["endog"] = dfxy["y"].values
            dresults[mdl_key]["exog"] = dfxy["y"].values
            dresults[mdl_key]["std_error"] = np.sqrt(mdl.mse_resid)
            dresults[mdl_key]["std_residuals"] =

            # Standard error of the regression: sqrt(MSE).
            std_error = np.sqrt(mdl.mse_resid)

            # mse_resid
            # self.ssr/self.df_resid

            # mse_model
            # self.ess/self.df_model



            # mdl.mse_resid




    def _regress_ldfs(self):
        """

        """


