"""
  _
 | |_ _ __(_)/| _(_) |
 | __| '__| | |/ / | __|
 | |_| |  | |   <| | |_
  \__|_|  |_|_|\_\_|\__|

A Pythonic Approach to Actuarial Reserving
Copyright 2018 James D. Triveri
"""
from functools import partial
from .datasets import dataref
from .triangle import totri
from .utils import _load, _load_lrdb, _get_datasets, _get_lrdb_lobs, _get_lrdb_specs


# Initialize dataset loading utility and lrdb-related functions.
lrdb_path = dataref["lrdb"]
load = partial(_load, dataref=dataref)
load_lrdb = partial(_load_lrdb, dataref=dataref)
get_datasets = partial(_get_datasets, dataref=dataref)
get_lrdb_lobs = partial(_get_lrdb_lobs, lrdb_path=lrdb_path)
get_lrdb_specs = partial(_get_lrdb_specs, lrdb_path=lrdb_path)

__version__ = '0.3.3'
