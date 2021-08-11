"""
  _
 | |_ _ __(_)/| _(_) |
 | __| '__| | |/ / | __|
 | |_| |  | |   <| | |_
  \__|_|  |_|_|\_\_|\__|

A Pythonic Approach to Actuarial Reserving
Copyright 2018 James D. Triveri




##### 0.2.11 Release TODO #####

 0. GLM reserve estimator
 1. Add additional Mack exhibits.
 2. Remove redundant triangle methods.
 3. Bootstrap residual diagnostic.
 5. Mack correlation test.
 4. MCMC reserve estimator
 8. Add airspeed velocity benchmarking classes.
 9. Write script to update version specified in __init__.py.
10. Update random seed initializer to new numpy format.
11. Verify incurred losses in lrdb have bulk removed.
"""
import collections
import datetime
from functools import partial
import os
import os.path
import sys
import warnings
import numpy as np
import pandas as pd
import scipy
from .datasets import dataref
from .triangle import totri
from .utils import (
    _load, _get_datasets, _get_lrdb_lobs, _get_lrdb_groups, _get_lrdb_specs,
    )

# Initialize dataset loading utility and lrdb-related functions.
lrdb_path = dataref["lrdb"]
load = partial(_load, dataref=dataref)
get_datasets = partial(_get_datasets, dataref=dataref)
get_lrdb_lobs = partial(_get_lrdb_lobs, lrdb_path=lrdb_path)
get_lrdb_groups = partial(_get_lrdb_groups, lrdb_path=lrdb_path)
get_lrdb_specs = partial(_get_lrdb_specs, lrdb_path=lrdb_path)

__version__ = "0.2.11"
