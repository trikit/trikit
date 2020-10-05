import unittest
import triangle_ut
import chainladder_ut
import utils_ut

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(triangle_ut))
suite.addTests(loader.loadTestsFromModule(chainladder_ut))
suite.addTests(loader.loadTestsFromModule(utils_ut))