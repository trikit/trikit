import sys; sys.path.append("G:\\Repos")


import numpy as np
import pandas as pd
import scipy
from scipy import stats
import trikit

from trikit.chainladder import BaseChainLadder
from trikit.chainladder.mack import MackChainLadder
from trikit.chainladder.bootstrap import BootstrapChainLadder




raa  = trikit.load(dataset="raa")
ta83 = trikit.load(dataset="ta83")
auto = trikit.load(dataset="lrdb", lob="COM_AUTO", grcode=32743)



tri = trikit.totri(data=raa)
cl_ = BaseChainLadder(tri)
cl = cl_.__call__()


mcl_ = MackChainLadder(tri)
mcl = mcl_.__call__()




bcl = BootstrapChainLadder(tri)


