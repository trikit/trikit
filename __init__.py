"""
<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>
|  _                                                                     |
| | |_ _ __(_) | _(_) |_                                                 |
| | __| '__| | |/ / | __|                                                |
| | |_| |  | |   <| | |_                                                 |
|  \__|_|  |_|_|\_\_|\__|                                                |
|                                                                        |
| Actuarial Reserving Methods in Python                                  |
|_______________________________________________________________________ |
|                                                                        |
| Created by      => James D Triveri <<<james.triveri@gmail.com>>>       |
| License         => 3-Clause BSD (See LICENSE in top-level directory    |
| Repository Link => https://github.com/jtrive84/trikit                  |                                                                  |
<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>=<>

Outstanding Tasks =>

[*] Find out how to suppress scientific notation without distorting the
     appearance of NaN's in DataFrames

[*] Implement BFChainLadder

[*] Implement MCMCChainLadder

[*] Add neg_handler #2 logic in _BootstrapChainLadder _bs_samples

[*] Add examples to *ChainLadder docstrings.

[*] Create trikit matplotlib stylesheet: https://matplotlib.org/users/style_sheets.html

[*] Accept additional styling arguments in triangler and chanladder plot method.

[*] Fix _BaseChainLadder's run method to mirror BootstrapChainLadder's run method.

[*] Update documentation for _ChainLadderResult

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

https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from .triangle import _IncrTriangle, _CumTriangle
from .chainladder import _BaseChainLadder
from .chainladder.mack import _MackChainLadder
from .chainladder.bootstrap import _BootstrapChainLadder
from .datasets import dataref
from .utils import (
    _load, _get_datasets, _get_lrdb_lobs, _get_lrdb_groups, _get_lrdb_specs,
    totri, _cumtoincr, _incrtocum, _tritotbl, _get_datasets #,plot_devp
    )

# Initialize `_load` closure using dataref mapping.
lrdb_path = dataref["lrdb"]
load = _load(dataref=dataref)
get_datasets = _get_datasets(dataref=dataref)
get_lrdb_lobs = _get_lrdb_lobs(lrdb_path=lrdb_path)
get_lrdb_groups = _get_lrdb_groups(lrdb_path=lrdb_path)
get_lrdb_specs = _get_lrdb_specs(lrdb_path=lrdb_path)

pd.options.mode.chained_assignment = None # 'warn'
#pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 500)
plt.style.use('ggplot')
np.set_printoptions(
    edgeitems=3, linewidth=200, suppress=True, nanstr='NaN',
    infstr='Inf', precision=5
    )

from .__pkginfo__ import version as __version__
from .__pkginfo__ import name



def chladder(data, origin=None, dev=None, value=None, trifmt=None,
             datafmt="incremental", range_method=None):
    """
    Return *ChainLadder initializer with optional reserve range method
    specification.

    Parameters
    ----------
    data: pd.DataFrame
        The dataset to be coerced into a *Triangle instance. ``data`` can be
        tabular loss data or a pandas DataFrame formatted as a triangle but
        not typed as such. In the latter case, ``trifmt`` must be specified to
        indicate the representation of ``data`` (either "cumulative" or
        "incremental").

    origin: str
        The field in ``data`` representing the origin year. When
        ``trifmt`` is not None, ``origin`` is ignored. Defaults to None.

    dev: str
        The field in ``data`` representing the development period. When
        ``trifmt`` is not None, ``dev`` is ignored. Defaults to None.

    value: str
        The field in ``data`` representing loss amounts. When ``trifmt`` is
        not None, ``value`` is ignored. Defaults to None.

    trifmt: str
        One of "cumulative", "incremental" or None (None by default).
        ``trifmt`` should only be set to something other than None if ``data``
        is a DataFrame formatted as a loss triangle, but hasn't yet been
        converted to a ``CumTriangle`` or ``IncrTriangle`` instance. When
        ``datafmt`` is not None, ``trifmt`` is ignored.

    datafmt: str
        When ``data`` is in tabular form, ``datafmt`` indicates whether the
        records comprising ``data`` represent cumulative or incremental
        losses.  When ``trifmt`` is not None, ``datafmt`` is ignored. Default
        value is "incremental".

    range_method: str
        One of "mack", "bootstrap" or None. Additional methods will
        be available in future releases. Defaults to None.

    Returns
    -------
    chainladder.*ChainLadder instance.
        An instance of chainladder.*ChainLadder, optionally with ranges
        quantifying reserve variability.

    Examples
    --------
    Here we demonstrate how to produce Chain Ladder estimates for the
    ``raa`` sample dataset. Note that what is returned by ``chladder``
    is callable: In order to obtain the Chain Ladder summary, the
    object returned by ``chladder`` must be called, optionally
    specifying ``sel`` (defaults to "all-weighted") and ``tail`` (defaults
    to 1.0). Implemented this way, the Chain Ladder ultimates resulting
    from any number of `loss development factor - tail factor` combinations
    can be specified for the same triangle and compared, in preference
    to instantiating a new ``_BaseChainLadder`` instance each time ``sel``
    or ``tail`` should be varied.

    >>> import trikit
    >>> data = trikit.load("ta83")
    >>> cl_init = trikit.chladder(data=data)
    >>> cl_init(sel="all-weighted", tail=1.005)
       origin maturity    latest      cldf       ultimate       reserve
    0    1981       10   18834.0  1.005000   18928.170000     94.170000
    1    1982        9   16704.0  1.014263   16942.243687    238.243687
    2    1983        8   23466.0  1.031441   24203.787778    737.787778
    3    1984        7   27067.0  1.065750   28846.657874   1779.657874
    4    1985        6   26180.0  1.110442   29071.370025   2891.370025
    5    1986        5   15852.0  1.236349   19598.608700   3746.608700
    6    1987        4   12314.0  1.448599   17838.049103   5524.049103
    7    1988        3   13112.0  1.841007   24139.288472  11027.288472
    8    1989        2    5395.0  2.988917   16125.209021  10730.209021
    9    1990        1    2063.0  8.964835   18494.454742  16431.454742
    10  total           160987.0       NaN  214187.839403  53200.839403

    """
    tri = totri(data=data, type_="cumulative", origin=origin, dev=dev,
                value=value, trifmt=trifmt, datafmt=datafmt)

    if range_method is None:
        # Instantiate BaseChainLadder instance.
        cl_init = _BaseChainLadder(cumtri=tri)

    elif range_method and range_method.lower().startswith("mack"):
        # Instantiate _MackChainLadder instance.
        cl_init = _MackChainLadder(cumtri=tri)

    elif range_method and range_method.lower().startswith("boot"):
        # Instantiate _BootstrapChainLadder instance.
        cl_init = _BootstrapChainLadder(cumtri=tri)

    else:
        raise NotImplementedError(
            "Specified range_method does not exist: `{}`".format(range_method)
            )

    return(cl_init)

