import sys; sys.path.append("E:\\Repos\\")
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



class TriangleFunctionsTestCase(unittest.TestCase):







class CumTriangleTestCase(unittest.TestCase):
    def setUp(self):
        self.ctri = totri(raa, type_="cumulative")

    def test_a2adim(self):
        self.assertEqual(
            self.ctri.shape[0]-1, self.ctri.a2a.shape[0],
            "Age-toage Factors not properly computed."
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
