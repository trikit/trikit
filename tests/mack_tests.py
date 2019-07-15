import sys; sys.path.append("G:\\Repos") #sys.path.append("C:\\Users\\cac9159\\Repos\\")
import unittest
import trikit
import pandas as pd
import numpy as np
import os
import os.path
import decimal
import logging
import timeit
import matplotlib.pyplot as plt
import seaborn as sns

from trikit.chainladder import mack

"""
### TODO ###
- Check that cl.ultimates == trisqrd[-1]
- Check that cl.trisqrd - cl.tri == 0
- Check that cl0.ultimates-cl0.reserves-cl0.latest == 0
- Review results when development periods are at increments other than year
"""

RAA = trikit.load("raa")
ta83 = trikit.load("ta83")

originlkp = {i:j for i, j in zip(range(1, 11), range(2001, 2011))}
devplkp = {i:j for i, j in zip(range(1, 11), range(12, 132, 12))}

ta83["_origin"] = ta83["origin"].map(originlkp)
ta83["_dev"] = ta83["dev"].map(devplkp)
ta83 = ta83.drop(["origin", "dev"], axis=1).rename({"_origin":"origin", "_dev":"dev"}, axis=1)



tri = trikit.totri(ta83)
mcl = mack.MackChainLadder(tri)

ldfs = mcl._ldfs(alpha=1, tail=1)
cldfs = mcl._cldfs(ldfs=ldfs)
ults = mcl._ultimates(cldfs=cldfs)
devpvar = mcl._devpvar(alpha=1.0, tail=1.0)
trisqrd = mcl._trisqrd(ldfs=ldfs)
dfratio = mcl._devpvar_ldf_ratio(ldfs=ldfs, devpvar=devpvar)
pelkp = mcl._index_reference()
proc_error = mcl._process_error(ldfs=ldfs, devpvar=devpvar)
param_error = mcl._parameter_error(ldfs=ldfs, devpvar=devpvar)











# ============================================================================]
class MackChainLadderTestCase(unittest.TestCase):
    def setUp(self):
        self.cl = trikit.BaseChainLadder(data=raa, sel="all-weighted", tail=1.0)


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






# Bootstrap Unit Tests ########################################################

class BootChainLadderTestCase(unittest.TestCase):
    def setUp(self):
        self.bcl = trikit.BootChainLadder(data=raa, sel="all-weighted", tail=1.0)


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
