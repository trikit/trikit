"""
<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>
|  _                                                                           |
| | |_ _ __(_) | _(_) |_                                                       |
| | __| '__| | |/ / | __|                                                      |
| | |_| |  | |   <| | |_                                                       |
|  \__|_|  |_|_|\_\_|\__|                                                      |
|                                                                              |
| Actuarial Reserving Methods in Python                                        |
|______________________________________________________________________________|
|                                                                              |
| Created by      : James D Triveri <<<james.triveri@gmail.com>>>              |
| License         : 3-Clause BSD (See LICENSE in top-level directory           |
| Repository Link : https://github.com/trikit/trikit.git                       |                                                                  |
<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>

Outstanding Tasks:

[*] Find out how to suppress scientific notation without distorting the
     appearance of NaN's in DataFrames

[*] Implement BFChainLadder

[*] Implement MCMCChainLadder

[*] Add neg_handler #2 logic in _BootstrapChainLadder _bs_samples

[*] Add examples to *ChainLadder docstrings.

[*] Create trikit matplotlib stylesheet: https://matplotlib.org/users/style_sheets.html

[*] Accept additional styling arguments in triangler and chanladder plot method.

[*] Fix _BaseChainLadder's run method to mirror BootstrapChainLadder's run method.

[*] Update documentation for _BaseChainLadderResult

[*] Fix names in triangle.origins/triangle.devp

[*] Add a non-standard triangle such that rows and columns are not identical.

[*] Fix this error:
# tri2 = DataFrame
trikit._IncrTriangle(tri2)
# Throws error:
# Traceback (most recent call last):
#   File "C:\Python37\lib\site-packages\IPython\core\interactiveshell.py", line 3296, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-75-5ef5b2b85c97>", line 1, in <module>
#     trikit._IncrTriangle(tri2)
#   File "G:\Repos\trikit\triangle.py", line 75, in __init__
#     for i in tri:
# UnboundLocalError: local variable 'tri' referenced before assignment
"""
import os
import sys
import collections
import datetime
import os.path
import warnings
import numpy as np
import pandas as pd
import scipy
from .datasets import dataref
from .triangle import totri
from .utils import (
    _load, _get_datasets, _get_lrdb_lobs, _get_lrdb_groups, _get_lrdb_specs,
    )


# Initialize load function using dataref mapping.
lrdb_path = dataref["lrdb"]
load = _load(dataref=dataref)
get_datasets = _get_datasets(dataref=dataref)
get_lrdb_lobs = _get_lrdb_lobs(lrdb_path=lrdb_path)
get_lrdb_groups = _get_lrdb_groups(lrdb_path=lrdb_path)
get_lrdb_specs = _get_lrdb_specs(lrdb_path=lrdb_path)

pd.options.mode.chained_assignment = None # 'warn'
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 500)
np.set_printoptions(
    edgeitems=3, linewidth=200, suppress=True, nanstr='NaN',
    infstr='Inf', precision=5
    )

# Bind reference to package version number.
version_path_ = os.path.dirname(__file__) + os.path.sep + "VERSION"
with open(version_path_) as fversion:
    __version__ = fversion.read().strip("\n")
