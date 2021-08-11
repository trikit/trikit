"""
Triangle benchmarks.
"""
import numpy as np
import pandas as pd
import trikit




class TriangleSuite:
    def setup(self):
        self.data_i = trikit.load(dataset="raa")
        self.tri_c = trikit.totri(data=self.data_i, tri_type="cumulative")
        self.tri_i = trikit.totri(data=self.data_i, tri_type="incr")
        self.dftri_cum = pd.DataFrame(self.tri_c)
        self.dftri_incr = pd.DataFrame(self.tri_i)
        self.data_c = self.tri_c.to_tbl()
        self.tri = self.tri_c

    def time_incr_tabular_to_incr(self):
        trikit.totri(data=self.data_i, tri_type="incremental", data_format="incr", data_shape="tabular")

    def time_incr_tabular_to_cum(self):
        trikit.totri(data=self.data_i, tri_type="cumulative", data_format="incr", data_shape="tabular")

    def time_cum_tabular_to_incr(self):
        trikit.totri(data=self.data_c, tri_type="incremental", data_format="cum", data_shape="tabular")

    def time_cum_tabular_to_cum(self):
        trikit.totri(data=self.data_c, tri_type="cumulative", data_format="cum", data_shape="tabular")

    def time_incr_trilike_to_incr(self):
        trikit.totri(data=self.dftri_incr, tri_type="incremental", data_format="incr", data_shape="triangle")

    def time_incr_trilike_to_cum(self):
        trikit.totri(data=self.dftri_incr, tri_type="cumulative", data_format="incr", data_shape="triangle")

    def time_cum_trilike_to_incr(self):
        trikit.totri(data=self.dftri_cum, tri_type="incremental", data_format="cum", data_shape="triangle")

    def time_cum_trilike_to_cum(self):
        trikit.totri(data=self.dftri_cum, tri_type="cumulative", data_format="cum", data_shape="triangle")

    def time_nbr_cells(self):
        self.tri.nbr_cells

    def time_triind(self):
        self.tri.triind

    def time_origins(self):
        self.tri.origins

    def time_devp(self):
        self.tri.devp

    def time_rlvi(self):
        self.tri.rlvi

    def time_clvi(self):
        self.tri.clvi

    def time_latest_by_origin(self):
        self.tri.latest_by_origin

    def time_latest_by_devp(self):
        self.tri.latest_by_devp

    def time_maturity(self):
        self.tri.maturity

    def time_diagonal_0(self):
        self.tri.diagonal(offset=0)

    def time_diagonal_1(self):
        self.tri.diagonal(offset=-1)

    def time_diagonal_k(self):
        self.tri.diagonal(offset=self.tri.devp.size-2)

    def time_totbl(self):
        self.tri.to_tbl()

    def time_to_incr(self):
        self.tri.to_incr()

    def time_a2a(self):
        self.tri.a2a

    def time_a2aind(self):
        self.tri.a2aind

    def time_a2a_avgs(self):
        self.tri.a2a_avgs()
