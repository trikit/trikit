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

RAA  = trikit.load("raa")
ta83 = trikit.load("ta83")
ctri = trikit.CumTriangle(data=RAA)


tri_init = pd.DataFrame({
    1 :[94,101,82,110,68,119,72,71,71,62],
    2 :[119,131,107,139,99,151,99,101,96,np.NaN],
    3 :[124,135,112,146,105,157,99,106,np.NaN,np.NaN],
    4 :[128,139,116,152,108,158,99,np.NaN,np.NaN,np.NaN],
    5 :[130,141,119,154,111,162,np.NaN,np.NaN,np.NaN,np.NaN],
    6 :[132,143,119,155,114,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN],
    7 :[133,143,120,156,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN],
    8 :[133,144,121,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN],
    9 :[133,145,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN],
    10:[134,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]
    }, index=range(1991,2001,1)
    )


tt = pd.DataFrame({
    1 :[94,101,82,110,68,119,72,71,71,62],
    2 :[119,131,107,139,99,151,99,101,96,np.NaN],
    3 :[124,135,112,146,105,157,99,106,np.NaN,np.NaN],
    4 :[128,139,116,152,108,158,99,np.NaN,np.NaN,np.NaN],
    5 :[130,141,119,154,111,162,np.NaN,np.NaN,np.NaN,np.NaN],
    6 :[132,143,119,155,114,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN],
    7 :[133,143,120,156,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN],
    8 :[133,144,121,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN],
    9 :[133,145,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN],
    10:[134,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN]
    }, index=range(1991,2001,1)
    )


# Instantiate MackChainLadder instance =======================================>
#btri = trikit.totri(data=tri_init, trifmt="cumulative", tritype="cumulative")
#bcl  = trikit.BootChainLadder(btri)


# ============================================================================>
bcl = trikit.BootChainLadder(RAA, sims=10, procdist="gamma", random_state=30)

nbrcells = bcl.nbr_cells
dof      = bcl.dof
sclparam = bcl.scale_param
trifitcum = bcl.tri_fit_cum
trifitincr = bcl.tri_fit_incr
resid_us    = bcl.resid_us
resid_adj   = bcl.resid_adj
sampd       = bcl.sampling_dist
bs_samps    = bcl.bs_samples
bs_ldfs     = bcl.bs_ldfs
bs_fore     = bcl.bs_forecasts
bspe        = bcl.bs_process_error(all_cols=False)


s = bcl.bootstrap_samples
l = bcl.bootstrap_ldfs
f = bcl.bootstrap_forecasts
