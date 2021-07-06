"""
Methods

assertEqual(a, b)
assertNotEqual(a, b)
assertTrue(x)
assertFalse(x)
assertIs(a, b)
assertIsNot(a, b)
assertIsNone(x)
assertIsNotNone(x)
assertIn(a, b)
assertNotIn(a, b)
assertIsInstance(a, b)
assertNotIsInstance(a, b)
"""
import unittest
import pandas as pd
import numpy as np
import os
import os.path
import logging
import timeit
import trikit



# BaseChainLadder -------------------------------------------------------------

class BaseChainLadderTestCase(unittest.TestCase):
    def setUp(self):
        data = trikit.load(dataset="raa")
        tri = trikit.totri(data, tri_type="cum", data_shape="tabular", data_format="incr")
        # cl = trikit.estimators.chainladder.BaseChainLadder(cumtri=tri)
        # r_cl = cl()
        r_cl = tri.base_cl()

        dactual_raa = {
            "ldfs_sum"     :13.28018030198903,
            "cldfs_sum"    :21.59861048771567,
            "latest_sum"   :321974.0,
            "ultimates_sum":213122.22826121017,
            "reserves_sum" :52135.228261210155,
            }

        self.ultimates_sum = r_cl.ultimates.drop("total").dropna().sum()
        self.reserves_sum = r_cl.reserves.drop("total").dropna().sum()
        self.latest_sum = r_cl.latest.dropna().sum()
        self.cldfs_sum = r_cl.cldfs.dropna().sum()
        self.ldfs_sum = r_cl.ldfs.dropna().sum()
        self.dactual_raa = dactual_raa


    def test_ldfs(self):
        # Test computed vs. reference LDF pattern.
        self.assertTrue(
            np.abs(self.ldfs_sum - self.dactual_raa["ldfs_sum"]) < 1.,
            "Non-equality between computed vs. reference LDFs."
            )

    def test_cldfs(self):
        # Test computed vs. reference LDF pattern.
        self.assertTrue(
            np.abs(self.cldfs_sum - self.dactual_raa["cldfs_sum"]) < 1.,
            "Non-equality between computed vs. reference CLDFs."
            )

    def test_latest(self):
        # Test computed vs. reference ultimates.
        self.assertTrue(
            np.abs(self.latest_sum - self.dactual_raa["latest_sum"]) < 1.,
            "Non-equality between computed vs. reference latest."
            )

    def test_ultimates(self):
        # Test computed vs. reference ultimates.
        self.assertTrue(
            np.abs(self.ultimates_sum - self.dactual_raa["ultimates_sum"]) < 1.,
            "Non-equality between computed vs. reference ultimates."
            )

    def test_reserves(self):
        # Test computed vs. reference ultimates.
        self.assertTrue(
            np.abs(self.reserves_sum - self.dactual_raa["reserves_sum"]) < 1.,
            "Non-equality between computed vs. reference reserves."
            )


# MackChainLadder -------------------------------------------------------------

class MackChainLadderTestCase(unittest.TestCase):

    def setUp(self):
        # Modify origin and development periods to test not sequentials.
        df = trikit.load(dataset="ta83")
        df["dev"] = df["dev"] * 12
        df["origin"] = df["origin"] + 2000
        tri = trikit.totri(df, tri_type="cum", data_shape="tabular", data_format="incr")
        mcl = trikit.chainladder.mack.MackChainLadder(cumtri=tri)
        r_lognorm = mcl(alpha=1, dist="lognorm")
        r_norm = mcl(alpha=1, dist="norm")

        dactual_ta83 = {
            "norm_mu_sum"        :18680869.054532073,
            "norm_sigma_sum"     :4771773.155719111,
            "norm_75_sum"        :21899381.138325423,
            "norm_95_sum"        :26529737.436706323,
            "lognorm_mu_sum"     :125.8539998696597,
            "lognorm_sigma_sum"  :2.6740386407158327,
            "lognorm_75_sum"     :21420867.75494642,
            "lognorm_95_sum"     :27371140.20920447,
            "mse_sum"            :4156154300629.1504,
            "std_error_sum"      :4771773.155719111,
            "cv_sum"             :2.80203003051732,
            "mse_total_sum"      :5989366778717.765,
            "process_error_sum"  :3527957849338.302,
            "parameter_error_sum":628196451290.8485,
            "ldfs_sum"           :14.207460332760107,
            "ultimates_sum"      :53038959.05453208,
            "reserves_sum"       :18680869.054532073,
            "devpvar_sum"        :279118.8961841563,
            "ldfvar_sum"         :0.05702584091389985,
            }

        with np.errstate(invalid="ignore"):
            self.norm_75_sum = pd.Series([
                r_norm.rvs[ii].ppf(.75) for ii in r_norm.tri.index
                ]).dropna().sum()
            self.norm_95_sum = pd.Series([
                r_norm.rvs[ii].ppf(.95) for ii in r_norm.tri.index
                ]).dropna().sum()
            self.lognorm_75_sum = pd.Series([
                r_lognorm.rvs[ii].ppf(.75) for ii in r_lognorm.tri.index
                ]).dropna().sum()
            self.lognorm_95_sum = pd.Series([
                r_lognorm.rvs[ii].ppf(.95) for ii in r_lognorm.tri.index
                ]).dropna().sum()

        self.mse_sum = r_lognorm.mse.dropna().sum()
        self.std_error_sum = r_lognorm.std_error.drop("total").dropna().sum()
        self.cv_sum = r_lognorm.cv.drop("total").dropna().sum()
        self.mse_total_sum = r_lognorm.mse_total.dropna().sum()
        self.process_error_sum = r_lognorm.process_error.dropna().sum()
        self.parameter_error_sum = r_lognorm.parameter_error.dropna().sum()
        self.ldfs_sum = r_lognorm.ldfs.dropna().sum()
        self.ultimates_sum = r_lognorm.ultimates.drop("total").dropna().sum()
        self.reserves_sum = r_lognorm.reserves.drop("total").dropna().sum()
        self.devpvar_sum = r_lognorm.devpvar.dropna().sum()
        self.ldfvar_sum = r_lognorm.ldfvar.dropna().sum()
        self.dactual_ta83 = dactual_ta83


    def test_ldfs(self):
        # Test computed vs. reference LDF pattern.
        self.assertEqual(
            self.ldfs_sum, self.dactual_ta83["ldfs_sum"],
            "Non-equality between computed vs. reference LDFs."
            )

    def test_ultimates(self):
        # Test computed vs. reference ultimates.
        self.assertTrue(
            np.abs(self.ultimates_sum - self.dactual_ta83["ultimates_sum"]) < 1.,
            "Non-equality between computed vs. reference ultimates."
            )

    def test_reserves(self):
        # Test computed vs. reference ultimates.
        self.assertTrue(
            np.abs(self.reserves_sum - self.dactual_ta83["reserves_sum"]) < 1.,
            "Non-equality between computed vs. reference reserves."
            )

    def test_devpvar(self):
        # Test computed vs. reference devpvar.
        self.assertTrue(
            np.abs(self.devpvar_sum - self.dactual_ta83["devpvar_sum"]) < 1.,
            "Non-equality between computed vs. reference devpvar."
            )

    def test_ldfvar(self):
        # Test computed vs. reference ldfvar.
        self.assertTrue(
            np.abs(self.ldfvar_sum - self.dactual_ta83["ldfvar_sum"]) < 1.,
            "Non-equality between computed vs. reference ldfvar."
            )

    def test_norm_75(self):
        # Test computed vs. reference normal 75th percentile of reserve distribution.
        self.assertTrue(
            np.abs(self.norm_75_sum - self.dactual_ta83["norm_75_sum"]) < 1.,
            "Non-equality between norm computed vs. reference 75th percentile."
            )

    def test_norm_95(self):
        # Test computed vs. reference normal 95th percentile of reserve distribution.
        self.assertTrue(
            np.abs(self.norm_95_sum - self.dactual_ta83["norm_95_sum"]) < 1.,
            "Non-equality between norm computed vs. reference 95th percentile."
            )

    def test_lognorm_75(self):
        # Test computed vs. reference log-normal 75th percentile of reserve distribution.
        self.assertTrue(
            np.abs(self.lognorm_75_sum - self.dactual_ta83["lognorm_75_sum"]) < 1.,
            "Non-equality between lognorm computed vs. reference 75th percentile."
            )

    def test_lognorm_95(self):
        # Test computed vs. reference log-normal 95th percentile of reserve distribution.
        self.assertTrue(
            np.abs(self.lognorm_95_sum - self.dactual_ta83["lognorm_95_sum"]) < 1.,
            "Non-equality between lognorm computed vs. reference 95th percentile."
            )

    def test_mse(self):
        # Test computed vs. reference aggregate mse.
        self.assertTrue(
            np.abs(self.mse_sum - self.dactual_ta83["mse_sum"]) < 1.,
            "Non-equality between computed vs. reference mse."
            )

    def test_std_error(self):
        # Test computed vs. reference aggregate std_error.
        self.assertTrue(
            np.abs(self.std_error_sum - self.dactual_ta83["std_error_sum"]) < 1.,
            "Non-equality between computed vs. reference std_error."
            )

    def test_cv(self):
        # Test computed vs. reference aggregate coefficient of variation.
        self.assertTrue(
            np.abs(self.cv_sum - self.dactual_ta83["cv_sum"]) < 1.,
            "Non-equality between computed vs. reference cv."
            )

    def test_mse_total(self):
        # Test computed vs. reference aggregate mse_total.
        self.assertTrue(
            np.abs(self.mse_total_sum - self.dactual_ta83["mse_total_sum"]) < 1.,
            "Non-equality between computed vs. reference mse_total."
            )

    def test_process_error(self):
        # Test computed vs. reference aggregate process error.
        self.assertTrue(
            np.abs(self.process_error_sum - self.dactual_ta83["process_error_sum"]) < 1.,
            "Non-equality between computed vs. reference process error."
            )

    def test_parameter_error(self):
        # Test computed vs. reference aggregate parameter error.
        self.assertTrue(
            np.abs(self.parameter_error_sum - self.dactual_ta83["parameter_error_sum"]) < 1.,
            "Non-equality between computed vs. reference parameter error."
            )



# BootstrapChainLadder tests --------------------------------------------------

class BootstrapChainLadderTestCase(unittest.TestCase):
    def setUp(self):
        df = trikit.load(dataset="raa")
        tri = trikit.totri(df, tri_type="cum", data_shape="tabular", data_format="incr")
        bcl = trikit.chainladder.bootstrap.BootstrapChainLadder(tri)
        r_bcl = bcl()


        dactual_raa = {
            "ldfs_sum"            :13.28018030198903,
            "cldfs_sum"           :21.59861048771567,
            "latest_sum"          :321974.0,
            "ultimates_sum"       :213122.22826121017,
            "reserves_sum"        :52135.228261210155,
            "dof"                 :56,
            "scale_param"         :632.3368030912758,
            "fitted_cum_sum"      :707622.0,
            "fitted_incr_sum"     :160987.0,
            "resid_us_sum"        :4.7274165831925234,
            "resid_adj_sum"       :4.68501737172304,
            "sampling_dist_sum"   :4.68501737172304,
            "bs_samples_sum"      :16198534.554200275,
            "bs_ldfs_sum"         :1245.7029695928195,
            "bs_forecasts_sum"    :21528898.436414644,
            "bs_process_error_sum":156451006.23791158,
            "bs_reserves_sum"     :5227886.663640983,
            }

        self.ultimates_sum = r_bcl.ultimates.drop("total").dropna().sum()
        self.reserves_sum = r_bcl.reserves.drop("total").dropna().sum()
        self.latest_sum = r_bcl.latest.dropna().sum()
        self.cldfs_sum = r_bcl.cldfs.dropna().sum()
        self.ldfs_sum = r_bcl.ldfs.dropna().sum()
        self.dof = r_bcl.dof
        self.tri_fit_cum = bcl._tri_fit_cum(r_bcl.ldfs)
        self.tri_fit_incr = bcl._tri_fit_incr(self.tri_fit_cum)
        self.resid_us = bcl._resid_us(self.tri_fit_incr)
        self.scale_param = bcl._scale_param(self.resid_us)
        self.resid_adj = bcl._resid_adj(self.resid_us)
        self.sampling_dist = bcl._sampling_dist(self.resid_adj)
        self.bs_samples = bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=100, random_state=516
            )
        self.bs_ldfs = bcl._bs_ldfs(self.bs_samples)
        self.bs_forecasts = bcl._bs_forecasts(
            self.bs_samples, self.bs_ldfs, self.scale_param
            )
        self.bs_process_error =  bcl._bs_process_error(
            self.bs_forecasts, self.scale_param, procdist="gamma", random_state=516
            )
        self.bs_reserves = bcl._bs_reserves(self.bs_process_error)
        self.dactual_raa = dactual_raa


    def test_ldfs(self):
        # Test computed vs. reference LDF pattern.
        self.assertTrue(
            np.abs(self.ldfs_sum - self.dactual_raa["ldfs_sum"]) < 1.,
            "Non-equality between computed vs. reference LDFs."
            )

    def test_cldfs(self):
        # Test computed vs. reference LDF pattern.
        self.assertTrue(
            np.abs(self.cldfs_sum - self.dactual_raa["cldfs_sum"]) < 1.,
            "Non-equality between computed vs. reference CLDFs."
            )

    def test_latest(self):
        # Test computed vs. reference ultimates.
        self.assertTrue(
            np.abs(self.latest_sum - self.dactual_raa["latest_sum"]) < 1.,
            "Non-equality between computed vs. reference latest."
            )

    def test_ultimates(self):
        # Test computed vs. reference ultimates.
        self.assertTrue(
            np.abs(self.ultimates_sum - self.dactual_raa["ultimates_sum"]) < 1.,
            "Non-equality between computed vs. reference ultimates."
            )

    def test_reserves(self):
        # Test computed vs. reference ultimates.
        self.assertTrue(
            np.abs(self.reserves_sum - self.dactual_raa["reserves_sum"]) < 1.,
            "Non-equality between computed vs. reference reserves."
            )


    def test_dof(self):
        # Test triangle degrees of freedom.
        self.assertEqual(
            self.dof, self.dactual_raa["dof"],
            "Non-equality between computed vs. reference degrees of freedom."
            )

    def test_scale_param(self):
        # Test computed vs. reference scale parameter.
        self.assertTrue(
            np.abs(self.scale_param - self.dactual_raa["scale_param"]) < 1.,
            "Non-equality between computed vs. reference scale parameter."
            )

    def test_tri_fit_cum(self):
        # Test computed vs. reference fitted cumulative triangle.
        self.assertEqual(
            self.tri_fit_cum.sum().sum(), self.dactual_raa["fitted_cum_sum"],
            "Non-equality between computed vs. reference cumulative triangle."
            )

    def test_tri_fit_incr(self):
        # Test computed vs. reference fitted incremental triangle.
        self.assertEqual(
            self.tri_fit_incr.sum().sum(), self.dactual_raa["fitted_incr_sum"],
            "Non-equality between computed vs. reference incremental triangle."
            )

    def test_resid_us(self):
        # Test computed vs. reference unscaled residuals.
        self.assertTrue(
            np.abs(self.resid_us.sum().sum() - self.dactual_raa["resid_us_sum"]) < 0.01,
            "Non-equality between computed vs. reference unscaled residuals."
            )

    def test_resid_adj(self):
        # Test computed vs. reference unscaled residuals.
        self.assertTrue(
            np.abs(self.resid_adj.sum().sum() - self.dactual_raa["resid_adj_sum"]) < 0.01,
            "Non-equality between computed vs. reference adjusted residuals."
            )

    def test_sampling_dist(self):
        # Test computed vs. reference sampling distribution.
        self.assertTrue(
            np.abs(self.sampling_dist.sum() - self.dactual_raa["sampling_dist_sum"]) < 0.01,
            "Non-equality between computed vs. reference sampling distribution."
            )

    def test_bs_samples(self):
        # Test bootstrap samples vs. reference aggregation.
        self.assertTrue(
            np.abs(self.bs_samples.samp_incr.dropna().sum() - self.dactual_raa["bs_samples_sum"]) < 1.0,
            "Non-equality between computed vs. reference bootstrapped samples."
            )

    def test_bs_ldfs(self):
        # Test bootstrap ldf samples vs. reference aggregation.
        self.assertTrue(
            np.abs(self.bs_ldfs.ldf.dropna().sum() - self.dactual_raa["bs_ldfs_sum"]) < 1.0,
            "Non-equality between computed vs. reference bootstrapped LDF samples."
            )

    def test_bs_forecasts(self):
        # Test bootstrapped forecasts vs. reference aggregation.
        self.assertTrue(
            np.abs(self.bs_forecasts.samp_incr.dropna().sum() - self.dactual_raa["bs_forecasts_sum"]) < 1.0,
            "Non-equality between computed vs. reference bootstrapped forecasts."
            )

    def test_bs_process_error(self):
        # Test bootstrapped process error vs. reference aggregation.
        self.assertTrue(
            np.abs(self.bs_process_error.ultimate.dropna().sum() - self.dactual_raa["bs_process_error_sum"]) < 1.0,
            "Non-equality between computed vs. reference bootstrapped process error."
            )

    def test_bs_reserves(self):
        # Test bootstrapped reserves vs. reference aggregation.
        self.assertTrue(
            np.abs(self.bs_reserves.reserve.dropna().sum() - self.dactual_raa["bs_reserves_sum"]) < 1.0,
            "Non-equality between computed vs. reference bootstrapped reserves."
            )


if __name__ == "__main__":

    unittest.main()