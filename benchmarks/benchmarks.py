# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import trikit


# class TimeSuite:
#     """
#     An example benchmark that times the performance of various kinds
#     of iterating over dictionaries in Python.
#     """
#     def setup(self):
#         self.d = {}
#         for x in range(500):
#             self.d[x] = None
#
#     def time_keys(self):
#         for key in self.d.keys():
#             pass
#
#     def time_iterkeys(self):
#         for key in self.d.iterkeys():
#             pass
#
#     def time_range(self):
#         d = self.d
#         for key in range(500):
#             x = d[key]
#
#     def time_xrange(self):
#         d = self.d
#         for key in xrange(500):
#             x = d[key]
#
#
# class MemSuite:
#     def mem_list(self):
#         return [0] * 256



class TriangleSuite:
    def setup(self):
        self.data = trikit.load(dataset="raa")

        self.tri = trikit.totri(data=data, type_="incremental")
        self.latest_ref = pd.DataFrame({
            "origin":list(range(1981, 1991, 1)), "maturity":list(range(10, 0, -1)),
            "dev":list(range(10, 0, -1)),
            "latest":[
                172.0, 535.0, 603.0, 984.0, 225.0, 2917.0, 1368.0,
                6165.0, 2262.0, 2063.0
            ],
        }, index=list(range(0, 10, 1))
        )





from .common import Benchmark

import numpy as np


class Core(Benchmark):
    def setup(self):
        self.l100 = range(100)
        self.l50 = range(50)
        self.float_l1000 = [float(i) for i in range(1000)]
        self.float64_l1000 = [np.float64(i) for i in range(1000)]
        self.int_l1000 = list(range(1000))
        self.l = [np.arange(1000), np.arange(1000)]
        self.l_view = [memoryview(a) for a in self.l]
        self.l10x10 = np.ones((10, 10))
        self.float64_dtype = np.dtype(np.float64)

    def time_array_1(self):
        np.array(1)

    def time_array_empty(self):
        np.array([])

    def time_array_l1(self):
        np.array([1])

    def time_array_l100(self):
        np.array(self.l100)

    def time_array_float_l1000(self):
        np.array(self.float_l1000)

    def time_array_float_l1000_dtype(self):
        np.array(self.float_l1000, dtype=self.float64_dtype)

    def time_array_float64_l1000(self):
        np.array(self.float64_l1000)

    def time_array_int_l1000(self):
        np.array(self.int_l1000)

    def time_array_l(self):
        np.array(self.l)

    def time_array_l_view(self):
        np.array(self.l_view)

    def time_vstack_l(self):
        np.vstack(self.l)

    def time_hstack_l(self):
        np.hstack(self.l)

    def time_dstack_l(self):
        np.dstack(self.l)

    def time_arange_100(self):
        np.arange(100)

    def time_zeros_100(self):
        np.zeros(100)

    def time_ones_100(self):
        np.ones(100)

    def time_empty_100(self):
        np.empty(100)

    def time_eye_100(self):
        np.eye(100)

    def time_identity_100(self):
        np.identity(100)

    def time_eye_3000(self):
        np.eye(3000)

    def time_identity_3000(self):
        np.identity(3000)

    def time_diag_l100(self):
        np.diag(self.l100)

    def time_diagflat_l100(self):
        np.diagflat(self.l100)

    def time_diagflat_l50_l50(self):
        np.diagflat([self.l50, self.l50])

    def time_triu_l10x10(self):
        np.triu(self.l10x10)

    def time_tril_l10x10(self):
        np.tril(self.l10x10)

    def time_triu_indices_500(self):
        np.triu_indices(500)

    def time_tril_indices_500(self):
        np.tril_indices(500)
