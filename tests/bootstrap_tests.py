"""
TODO:

Check that initializing triangle instance with fields other than
origin, dev, value still works with BootstrapChainLadder.

Incorporate unit tests.

"""
import sys
import platform
if platform.node()=="PC0VPZB1":
    sys.path.append("C:\\Users\\cac9159\\Repos")
else:
    sys.path.append("G:\\Repos")


import unittest
import trikit
import pandas as pd
import numpy as np
import os
import os.path
import decimal
import logging
import timeit
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import trikit


RAA  = trikit.load("raa")
ta83 = trikit.load("ta83")

# ============================================================================]
tri = trikit.totri(RAA)
cl  = tri.chladder(sel="all-weighted", tail=1.0)
bcl = tri.chladder(range_method="bootstrap", sims=1000)



d0 = bcl.origin_distribution
d1 = bcl.aggregate_distribution




bcl.plot(which="reserve", q=.9)
bcl.plot(which="ultimate", q=.95)






# ============================================================================]
# bcl = tri.chladder(range_method="bootstrap", q=[.75, .95], symmetric=False, sims=1000)
# tfc = bcl._tri_fit_cum(sel="all-weighted")
# tfi = bcl._tri_fit_incr(tfc)
# r_us = bcl._resid_us(tfi)
# sclp = bcl._scale_param(r_us)
# r_adj = bcl._resid_adj(r_us)
# sampd = bcl._sampling_dist(r_adj)
# dfsamples = bcl._bs_samples(sampd, tfi, sims=100)
# dfldfs = bcl._bs_ldfs(dfsamples)
# dflvi = bcl.tri.rlvi.reset_index(drop=False)
# dflvi = dflvi.rename({"index":"origin", "dev":"l_act_dev"}, axis=1)
# dflvi = dflvi.drop("col_offset", axis=1)
# dfcombined = dfsamples.merge(dfldfs, on=["sim", "dev"], how="left")
# dfcombined = dfcombined.merge(dflvi, on=["origin"], how="left", )
# dfcombined = dfcombined.reset_index(drop=True).sort_values(by=["sim", "origin", "dev"])
# dfforecasts = bcl._bs_forecasts(dfcombined=dfcombined, scale_param=sclp)
# dfprocerror = bcl._bs_process_error(dfforecasts=dfforecasts, scale_param=sclp,)
# ============================================================================]





# Bootstrap Unit Tests ########################################################

class BootstrapChainLadderTestCase(unittest.TestCase):
    def setUp(self):

        clkwds = {"sims":1000, "neg_handler":1, "procdist":"gamma",
                  "parametric":False, "q":[.75, .95], "symmetric":True,
                  "interpolation":"linear",}

        self.tri = trikit.totri(trikit.load("raa"))
        self.bcl = self.tri.chladder(range_method="bootstrap", **clkwds)





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
