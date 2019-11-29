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
import sys; sys.path.append("G:\\Repos\\")
import unittest
import trikit
import pandas as pd
import numpy as np
import os
import os.path
import logging
import timeit


from ..triangle import incremental, cumulative
from ..utils import _load, totri, _cumtoincr, _incrtocum, _tritotbl
from ..datasets import dataref

load = _load(dataref)


raa  = trikit.load(dataset="raa")
ta83 = trikit.load(dataset="ta83")
auto = trikit.load(dataset="lrdb", loss_key="COM_AUTO", grcode=32743)
nstd = trikit.load(dataset="nonstd")

ctri = totri(data=raa, datatype="incremental", tritype="cumulative")


# check that s.split fails when the separator is not a string
# with self.assertRaises(TypeError):
    # s.split(2)



class IncrTriangleTestCase(unittest.TestCase):
    def setUp(self):
        self.tri = trikit.totri(data=raa, type_="incremental")
        self.latest_ref = pd.DataFrame({
            "origin":list(range(1981, 1991, 1)),
            "dev":list(range(10, 0, -1)),
            "latest":[172.0, 535.0, 603.0, 984.0, 225.0, 2917.0, 1368.0,
                      6165.0, 2262.0, 2063.0],
            }, index=list(range(0, 10, 1))
            )


    def test_nbr_cells(self):
        self.assertEqual(
            self.tri.nbr_cells, 55
            )

    def test_dof(self):
        self.assertEqual(
            self.tri.dof, 56
            )

    def test_triind(self):
        triindprod_ = (self.tri.triind * self.tri).sum().sum()
        self.assertTrue(
            np.allclose(triindprod, 0)
            )

    def test_rlvi(self):
        ref_ = pd.DataFrame({
            "dev":list(range(10, 0, -1)),
            "col_offset":list(range(9, -1, -1)),
            }, index=list(range(1981, 1991, 1))
            ).sort_index()
        self.assertTrue(
            ref_.equals(self.tri.rlvi)
            )

    def test_clvi(self):
        ref_ = pd.DataFrame({
            "origin":list(range(1990, 1980, -1)),
            "row_offset":list(range(9, -1, -1)),
            }, index=list(range(1, 11, 1))
            ).sort_index()
        self.assertTrue(
            ref_.equals(self.tri.clvi)
            )

    def test_devp(self):
        ref_ = pd.Series(
            data=self.latest_ref.dev.values.tolist()[::-1],
            name="devp"
            ).sort_index()
        self.assertTrue(
            ref_.equals(self.tri.devp)
            )

    def test_origins(self):
        ref_ = pd.Series(
            data=self.latest_ref.origin.values.tolist(),
            name="origin"
            ).sort_index()
        self.assertTrue(
            ref_.equals(self.tri.origins)
            )

    def test_maturity(self):
        ref_ = pd.Series(
            data=self.latest_ref.dev.values.tolist(),
            index=self.latest_ref.origin.values.tolist(),
            name="maturity"
            ).sort_index()
        self.assertTrue(
            ref_.equals(self.tri.origins)
            )

    def test_latest(self):
        self.assertTrue(
            self.latest_ref.equals(self.tri.latest)
            )

    def test_latest_by_origin(self):
        ref_ = pd.Series(
            data=self.latest_ref.latest,
            index=self.latest_ref.origin,
            name="latest_by_origin"
            ).sort_index()
        self.assertTrue(
            ref_.equals(self.tri.latest_by_origin)
            )

    def test_latest_by_devp(self):
        ref_ = pd.Series(
            data=self.latest_ref.latest,
            index=self.latest_ref.dev,
            name="latest_by_devp"
            ).sort_index()
        self.assertTrue(
            ref_.equals(self.tri.latest_by_devp)
            )

    def test_as_tbl(self):
        pass

    def test_as_cum(self):
        pass

    def test_as_incr(self):
        pass




# =============================================================================
# CumTriangle Tests
# =============================================================================

class CumTriangleTestCase(unittest.TestCase):
    def setUp(self):
        self.ctri = totri(raa, type_="cumulative")

    def test_a2adim(self):
        self.assertEqual(
            self.ctri.shape[0]-1, self.ctri.a2a.shape[0],
            "Age-to-age Factors not properly computed."
            )

    def test_latest_by_origin(self):
        self.assertEqual(
            self.ctri.shape[0]-1, self.ctri.a2a.shape[0],
            "Age-toage Factors not properly computed."
            )





#
class ChainLadderTestCase(unittest.TestCase):
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
