import unittest
import triangle_ut
import chainladder_ut
import datasets_ut

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(triangle_ut))
suite.addTests(loader.loadTestsFromModule(chainladder_ut))
suite.addTests(loader.loadTestsFromModule(datasets_ut))
