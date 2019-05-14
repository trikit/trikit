import sys; sys.path.append("E:\\Repos\\")
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

from trikit.triangle import (
    IncrTriangle,
    CumTriangle,
    _cumtoincr,
    _incrtocum,
    _tritotbl,
    plot_devp
    )

"""
 ### TODO ###
- Check that cl.ultimates == trisqrd[-1]
- Check that cl.trisqrd - cl.tri == 0
- Check that cl0.ultimates-cl0.reserves-cl0.latest == 0
"""

RAA  = trikit.load("raa")
ta83 = trikit.load("ta83")
ctri = trikit.CumTriangle(data=RAA)



# Instantiate MackChainLadder instance =======================================>
mcl0 = trikit.MackChainLadder(data=ta83, alpha=1, tail=1)
mcl1 = trikit.MackChainLadder(data=RAA, alpha=1, tail=1)

ldfs    = mcl0.ldfs
cldfs   = mcl0.cldfs
invsums = mcl0.inverse_sums
devpvar  = mcl0.devpvar
procerr = mcl0.process_error
originref = mcl0.originref
devpref = mcl0.devpref
rmse    = mcl0.rmsepi

def f(*args):
    return([i for i in args])
