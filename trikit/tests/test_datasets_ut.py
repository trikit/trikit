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



# Ensure sample datasets have been properly loaded ----------------------------

class DatasetsTestCase(unittest.TestCase):

    def setUp(self):
        self.dactual = {
            "raa"         :160987,
            "ta83"        :34358090,
            "autoliab"    :2197134,
            "glre"        :56531053,
            "singinjury"  :11026482.299999999,
            "singproperty":25101206.26,
            "lrdb"        :10178930,
            }

    def test_raa(self):
        self.assertEqual(
            trikit.load("raa").value.sum(), self.dactual["raa"],
            "Issue detected with raa sample dataset."
            )

    def test_ta83(self):
        self.assertEqual(
            trikit.load("ta83").value.sum(), self.dactual["ta83"],
            "Issue detected with ta83 sample dataset."
            )

    def test_autoliab(self):
        self.assertEqual(
            trikit.load("autoliab").value.sum(), self.dactual["autoliab"],
            "Issue detected with autoliab sample dataset."
            )

    def test_glre(self):
        self.assertEqual(
            trikit.load("glre").value.sum(), self.dactual["glre"],
            "Issue detected with glre sample dataset."
            )

    def test_singinjury(self):
        self.assertEqual(
            trikit.load("singinjury").value.sum(), self.dactual["singinjury"],
            "Issue detected with singinjury sample dataset."
            )

    def test_singproperty(self):
        self.assertEqual(
            trikit.load("singproperty").value.sum(), self.dactual["singproperty"],
            "Issue detected with singproperty sample dataset."
            )

    def test_raa(self):
        self.assertEqual(
            trikit.load("raa").value.sum(), self.dactual["raa"],
            "Issue detected with raa sample dataset."
            )

    def test_lrdb(self):
        self.assertEqual(
            trikit.load("lrdb").value.sum(), self.dactual["lrdb"],
            "Issue detected with lrdb sample dataset."
            )
