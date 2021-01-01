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




class ChainLadderTestCase(unittest.TestCase):
    def setUp(self):
        data = trikit.load(dataset="raa")
        tri = trikit.totri(data, type_="cum", data_shape="tabular", data_format="incr")
        self.cl = trikit.chainladder.BaseChainLadder(cumtri=tri).__call__(sel="all-weighted", tail=1.)

        raa_cl_ref = pd.DataFrame({
            "origin":[1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990,],
            "maturity":['10', '9', '8', '7', '6', '5', '4', '3', '2', '1',],
            "cldf":[1., 1.00922, 1.02631, 1.06045, 1.10492, 1.2302 , 1.44139, 1.83185, 2.97405, 8.92023,],
            "latest":[ 18834.,  16704.,  23466.,  27067.,  26180.,  15852.,  12314.,  13112.,   5395.,   2063.,],
            "ultimate":[ 18834.,  16857.95392,  24083.37092,  28703.14216,  28926.73634,  19501.10318,  17749.30259,
                         24019.19251,  16044.9841 ,  18402.44253,],
            "reserve":[0.,   153.95392,   617.37092,  1636.14216,  2746.73634,  3649.10318,  5435.30259, 10907.19251,
                       10649.9841 , 16339.44253,]
            })

        ref_ldfs = pd.Series(
            [2.99936, 1.62352, 1.27089, 1.17167, 1.11338, 1.04193, 1.03326, 1.01694, 1.00922, 1.,],
            dtype=np.float
            )

        ref_cldfs =  np.asarray(
            [8.92023, 2.97405, 1.83185, 1.44139, 1.2302 , 1.10492, 1.06045, 1.02631, 1.00922, 1.],
            dtype=np.float
            )

        self.raa_cl_ref = raa_cl_ref#[raa_cl_ref.index!="total"]
        self.ref_ldfs = ref_ldfs
        self.ref_cldfs = ref_cldfs



    def test_sel_ldfs(self):
        # Test computed vs. reference selected LDF pattern.
        refldfs = pd.Series(data=self.ref_ldfs).to_frame().reset_index(drop=False).rename({"index":"dev", 0:"ldf0"}, axis=1)
        refldfs["dev"] = refldfs["dev"].map(lambda v: v + 1)
        clldfs = self.cl.ldfs.to_frame().reset_index(drop=False).rename({"index":"dev", "ldf":"ldf1"}, axis=1)
        df = refldfs.merge(clldfs, on="dev", how="left")
        df["diff"] = df["ldf1"] - df["ldf0"]
        self.assertTrue(
            np.abs(df["diff"].sum().astype(np.float))<.01,
            "Non-equality between computed vs. reference LDFs."
            )

    def test_sel_cldfs(self):
        # Test computed vs. reference selected LDF pattern.
        refcldfs = pd.Series(data=self.ref_cldfs).to_frame().reset_index(drop=False).rename({"index":"dev", 0:"cldf0"}, axis=1)
        refcldfs["dev"] = refcldfs["dev"] + 1
        clcldfs = self.cl.cldfs.to_frame().reset_index(drop=False).rename({"index":"dev", "cldf":"cldf1"}, axis=1)
        df = refcldfs.merge(clcldfs, on="dev", how="left")
        df["diff"] = df["cldf1"] - df["cldf0"]
        self.assertTrue(
            df["diff"].sum().astype(np.float)<.01,
            "Non-equality between computed vs. reference CLDFs."
            )

    def test_ultimates(self):
        # Verify that computed ultimates match raa_cl_ref.ultimate.
        clults = self.cl.ultimates.reset_index(drop=False).rename(
            {"index":"origin", "ultimate":"ultimate1"}, axis=1
            )
        refults = self.raa_cl_ref[["origin", "ultimate"]].rename(
            {"ultimate":"ultimate0"}, axis=1
            )
        df = refults.merge(clults, on="origin", how="left")
        df["diff"] = df["ultimate1"] - df["ultimate0"]
        self.assertTrue(
            df["diff"].sum().astype(np.float)<1.,
            "Difference in computed vs. reference ultimates."
            )

    def test_trisqrd(self):
        # Verify that ultimates matches the last column of trisqrd.
        refults = self.raa_cl_ref[["origin", "ultimate"]].rename(
            {"ultimate":"ultimate0"}, axis=1
            )
        tsults = self.cl.trisqrd.loc[:,self.cl.trisqrd.columns[-1]].to_frame().reset_index(drop=False).rename(
            {"index":"origin", "ultimate":"ultimate1"}, axis=1
            )
        df = refults.merge(tsults, on="origin", how="left")
        df["diff"] = df["ultimate1"] - df["ultimate0"]
        self.assertTrue(
            df["diff"].sum().astype(np.float)<1.,
            "Difference in cl.trisqrd vs. reference ultimates."
            )

    def test_reserves(self):
        # Verify that computed reserves match raa_cl_ref.reserve.
        refres = self.raa_cl_ref[["origin", "reserve"]].rename(
            {"reserve":"reserve0"}, axis=1
            )
        clres = self.cl.reserves.reset_index(drop=False).rename(
            {"index":"origin", "reserve":"reserve1"}, axis=1
            )
        df = refres.merge(clres, on="origin", how="left")
        df["diff"] = df["reserve1"] - df["reserve0"]
        self.assertTrue(
            np.abs(df["diff"].sum())<1,
            "Difference in computed vs. reference reserves."
            )



# MackChainLadder.
# Test developemnt/origin periods other than 1-10.

# class MackChainLadderTestCase(unittest.TestCase):
#     def setUp(self):
#         data = trikit.load(dataset="ta83")
#         tri = trikit.totri(data, type_="cum", data_shape="tabular", data_format="incr")
#         self.mcl = trikit.chainladder.mack.MackBaseChainLadder(cumtri=tri).__call__(alpha=1)
#
#         raa_cl_ref = pd.DataFrame({
#             "origin":[1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990,],
#             "maturity":['10', '9', '8', '7', '6', '5', '4', '3', '2', '1',],
#             "cldf":[1., 1.00922, 1.02631, 1.06045, 1.10492, 1.2302 , 1.44139, 1.83185, 2.97405, 8.92023,],
#             "latest":[ 18834.,  16704.,  23466.,  27067.,  26180.,  15852.,  12314.,  13112.,   5395.,   2063.,],
#             "ultimate":[ 18834.,  16857.95392,  24083.37092,  28703.14216,  28926.73634,  19501.10318,  17749.30259,
#                          24019.19251,  16044.9841 ,  18402.44253,],
#             "reserve":[0.,   153.95392,   617.37092,  1636.14216,  2746.73634,  3649.10318,  5435.30259, 10907.19251,
#                        10649.9841 , 16339.44253,]
#         })
#
#         ref_ldfs = pd.Series(
#             [2.99936, 1.62352, 1.27089, 1.17167, 1.11338, 1.04193, 1.03326, 1.01694, 1.00922, 1.,],
#             dtype=np.float
#         )
#
#         ref_cldfs =  np.asarray(
#             [8.92023, 2.97405, 1.83185, 1.44139, 1.2302 , 1.10492, 1.06045, 1.02631, 1.00922, 1.],
#             dtype=np.float
#         )
#
#         self.raa_cl_ref = raa_cl_ref#[raa_cl_ref.index!="total"]
#         self.ref_ldfs = ref_ldfs
#         self.ref_cldfs = ref_cldfs
#
#         self.dactual = {}











# BootstrapChainLadder tests --------------------------------------------------

class BootstrapChainLadderTestCase(unittest.TestCase):
    def setUp(self):
        data = trikit.load(dataset="raa")
        tri = trikit.totri(data, type_="cum", data_shape="tabular", data_format="incr")
        self.bcl = trikit.BootChainLadder(data=raa, sel="all-weighted", tail=1.0)

        raa_cl_ref = pd.DataFrame({
            "origin":[1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990,],
            "maturity":['10', '9', '8', '7', '6', '5', '4', '3', '2', '1',],
            "cldf":[1., 1.00922, 1.02631, 1.06045, 1.10492, 1.2302 , 1.44139, 1.83185, 2.97405, 8.92023,],
            "latest":[ 18834.,  16704.,  23466.,  27067.,  26180.,  15852.,  12314.,  13112.,   5395.,   2063.,],
            "ultimate":[ 18834.,  16857.95392,  24083.37092,  28703.14216,  28926.73634,  19501.10318,  17749.30259,
                         24019.19251,  16044.9841 ,  18402.44253,],
            "reserve":[0.,   153.95392,   617.37092,  1636.14216,  2746.73634,  3649.10318,  5435.30259, 10907.19251,
                       10649.9841 , 16339.44253,]
            })

        ref_ldfs = pd.Series(
            [2.99936, 1.62352, 1.27089, 1.17167, 1.11338, 1.04193, 1.03326, 1.01694, 1.00922, 1.,],
            dtype=np.float
            )

        ref_cldfs =  np.asarray(
            [8.92023, 2.97405, 1.83185, 1.44139, 1.2302 , 1.10492, 1.06045, 1.02631, 1.00922, 1.],
            dtype=np.float
            )
        
        self.dactual_raa = {
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

        self.raa_cl_ref = raa_cl_ref#[raa_cl_ref.index!="total"]
        self.ref_ldfs = ref_ldfs
        self.ref_cldfs = ref_cldfs
        self.tri_fit_cum = self.bcl._tri_fit_cum(self.ref_ldfs)
        self.tri_fit_incr = self.bcl._tri_fit_incr(self.tri_fit_cum)
        self.resid_us = self.bcl._resid_us(self.tri_fit_incr)
        self.scale_param = self.bcl._scale_param(self.resid_us)
        self.resid_adj = self.bcl._resid_adj(self.resid_us)
        self.sampling_dist = self.bcl._sampling_dist(self.resid_adj)
        self.bs_samples = self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=100, random_state=516
            )
        self.bs_ldfs = self.bcl._bs_ldfs(self.bs_samples)
        self.bs_forecasts = self.bcl._bs_forecasts(
            self.bs_samples, self.bs_ldfs, self.scale_param
            )
        self.bs_process_error =  self.mcl._bs_process_error(
            self.bs_forecasts, self.scale_param, procdist="gamma", random_state=516
            )
        self.bs_reserves = self.bcl._bs_reserves(self.bs_process_error)


    def test_dof(self):
        # Test triangle degrees of freedom.
        self.assertEqual(
            self.bcl.dof, self.dactual_raa["fitted_cum_sum"],
            "Non-equality between computed vs. reference degrees of freedom."
            )

    def test_scale_param(self):
        # Test computed vs. reference scale parameter.
        self.assertTrue(
            np.abs(self.scale_param - self.dactual_raa["scale_param"]) < 1.,
            "Non-equality between computed vs. reference scale parameter."
            )

    def test_sel_ldfs(self):
        # Test computed vs. reference selected LDF pattern.
        refldfs = pd.Series(data=self.ref_ldfs).to_frame().reset_index(drop=False).rename({"index":"dev", 0:"ldf0"}, axis=1)
        refldfs["dev"] = refldfs["dev"].map(lambda v: v + 1)
        clldfs = self.cl.ldfs.to_frame().reset_index(drop=False).rename({"index":"dev", "ldf":"ldf1"}, axis=1)
        df = refldfs.merge(clldfs, on="dev", how="left")
        df["diff"] = df["ldf1"] - df["ldf0"]
        self.assertTrue(
            np.abs(df["diff"].sum().astype(np.float))<.01,
            "Non-equality between computed vs. reference LDFs."
            )

    def test_sel_cldfs(self):
        # Test computed vs. reference selected LDF pattern.
        refcldfs = pd.Series(data=self.ref_cldfs).to_frame().reset_index(drop=False).rename({"index":"dev", 0:"cldf0"}, axis=1)
        refcldfs["dev"] = refcldfs["dev"] + 1
        clcldfs = self.bcl.cldfs.to_frame().reset_index(drop=False).rename({"index":"dev", "cldf":"cldf1"}, axis=1)
        df = refcldfs.merge(clcldfs, on="dev", how="left")
        df["diff"] = df["cldf1"] - df["cldf0"]
        self.assertTrue(
            df["diff"].sum().astype(np.float)<.01,
            "Non-equality between computed vs. reference CLDFs."
            )

    def test_tri_fit_cum(self):
        # Test computed vs. reference fitted cumulative triangle.
        # tri_fit_cum_sum = self.bcl._tri_fit_cum(self.ref_ldfs).sum().sum()
        self.assertTrue(
            np.abs(self.tri_fit_cum.sum().sum() - self.dactual_raa["fitted_cum_sum"]) < 1.,
            "Non-equality between computed vs. reference cumulative triangle."
            )

    def test_tri_fit_incr(self):
        # Test computed vs. reference fitted incremental triangle.
        # tri_fit_cum = self.bcl._tri_fit_cum(self.ref_ldfs)
        # tri_fit_incr_sum = self.bcl._tri_fit_incr(tri_fit_cum).sum().sum()
        self.assertTrue(
            np.abs(self.tri_fit_incr.sum().sum() - self.dactual_raa["fitted_incr_sum"]) < 1.,
            "Non-equality between computed vs. reference incremental triangle."
            )

    def test_resid_us(self):
        # Test computed vs. reference unscaled residuals.
        self.assertTrue(
            np.abs(self.resid_us.sum().sum() - self.dactual_raa["resid_us_sum"]) < 0.01,
            "Non-equality between computed vs. reference unscaled residuals."
            )

    def test_resid_adj():
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





















    tri_fit_incr = self._tri_fit_incr(fitted_tri_cum=tri_fit_cum)


    def test_trisqrd(self):
        """
        Verify that self.tri and self.trisqrd are the same in
        upper left.
        """
        self.assertEqual(
            (self.cl.trisqrd-self.cl.tri).sum().sum(),0.0,
            "trisqrd not correctly implemented."
            )


    def test_ultimates(self):
        """
        Verify that ultimates matches the last column of trisqrd.
        """
        atults = self.cl.ultimates
        tsults = self.cl.trisqrd.loc[:,self.cl.trisqrd.columns[-1]]
        self.assertEqual(
            atults, tsults, "Difference in ultimate results"
            )

    def test_reserves(self):
        """
        Test value consistency.
        """
        assertEqual(
            (self.cl.ultimates-self.cl.latest_by_origin-self.cl.reserves).sum(),
            0, "Inconsistency in ults, latest and reserves."
            )


if __name__ == "__main__":

    unittest.main()