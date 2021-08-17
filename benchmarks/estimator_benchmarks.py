"""
Reserve estimator benchmarks.
"""
import sys
sys.path.append("G:\\Repos\\_trikit_\\trikit")
import numpy as np
import pandas as pd
import trikit
from trikit.estimators import chainladder


class BaseChainLadderSuite:
        def setup(self):
            data = trikit.load(dataset="raa")
            self.tri = trikit.totri(data=data, type_="cumulative")
            self.cl = chainLadder.BaseChainLadder(self.tri)
            self.ldfs = self.cl._ldfs(sel="all-weighted")
            self.cldfs = self.cl._cldfs(self.ldfs)
            self.ults = self.cl._reserves(self.ultimates)
            self.clr = self.cl.__call__()

        def time_ldfs(self):
            self.cl._ldfs()

        def time_cldfs(self):
            self.cl._cldfs(ldfs=self.ldfs)

        def time_ultimates(self):
            self.cl._ultimates(self.cldfs)

        def time_reserves(self):
            self.cl._reserves(self.ults)

        def time_trisqrd(self):
            self.cl._trisqrd(self.ldfs)

        def time___call__(self):
            self.tri.base_cl()

        def time_get_y_ticks(self):
            self.clr._get_yticks()

        def time_clr_data_transform(self):
            self.clr._data_transform()

        def time_base_cl(self):
            self.cl.__call__(sel="all-weighted", tail=1.0)

        def time_base_cl_tail(self):
            self.cl.__call__(sel="all-weighted", tail=1.05)



class BootstrapChainLadderSuite:
    def setup(self):
        data = trikit.load(dataset="raa")
        self.tri = trikit.totri(data=data, type_="cumulative")
        self.bcl = chainLadder.BootstrapChainLadder(self.tri)
        self.qtls, self.qtlhdrs = self.bcl._qtls_formatter(q=[.75, .95], two_sided=True)
        self.ldfs = self.bcl._ldfs(sel="all-weighted")
        self.cldfs = self.bcl._cldfs(self.ldfs)
        self.ults = self.bcl._reserves(self.ultimates)
        self.clr = self.bcl.__call__()
        self.tri_fit_cum = self.bcl._tri_fit_cum(self.ldfs)
        self.tri_fit_incr = self.bcl._tri_fit_incr(self.tri_fit_cum)
        self.resid_us = self.bcl._resid_us(self.tri_fit_incr)
        self.scale_param = self.bcl._scale_param(self.resid_us)
        self.resid_adj = self.bcl._resid_adj(self.resid_us)
        self.sampling_dist = self.bcl._sampling_dist(self.resid_adj)
        self.dfsamples = self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=1000, parametric=False,
            random_state=516
            )
        self.dfldfs = self.bcl._bs_ldfs(self.dfsamples)
        self.dfforecasts = self.bcl._bs_forecasts(self.dfsamples, self.dfldfs, self.scale_param)
        self.dfprocess_error = self.bcl._bs_process_error(
            self.dfforecasts, scale_param, procdist="gamma", random_state=516
            )
        self.dfreserves = self.bcl._bs_reserves(self.dfprocess_error)
        self.bclr = self.bcl.__call__()
        self.seed = 516

    def time_qtls_formatter(self):
        self.bcl._qtls_formatter(q=[.75, .95], two_sided=True)

    def time_dfrlvi(self):
        self.bcl.dfrlvi

    def time_get_dfcombined(self):
        self.bcl._get_dfcombined(self.dfsamples, self.dfldfs)

    def time_dof(self):
        self.bcl.dof

    def time_scale_param(self):
        self.bcl._scale_param(self.resid_us)

    def time_tri_fit_cum(self):
        self.bcl._tri_fit_cum(self.ldfs)
        
    def time_tri_fit_incr(self):
        self.bcl._tri_fit_incr(self.tri_fit_cum)
        
    def time_resid_us(self):
        self.bcl._resid_us(self.tri_fit_incr)
        
    def time_resid_adj(self):
        self.bcl._resid_adj(self.resid_us)
        
    def time_sampling_dist(self):
        self.bcl._sampling_dist(self.resid_adj)

    def time_bs_samples_500(self):
        self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=500, parametric=False,
            random_state=self.seed
            )

    def time_bs_samples_1000(self):
        self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=1000, parametric=False, 
            random_state=self.seed
            )

    def time_bs_samples_2500(self):
        self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=2500, parametric=False,
            random_state=self.seed
            )

    def time_bs_samples_5000(self):
        self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=5000, parametric=False,
            random_state=self.seed
            )

    def time_bs_samples_10000(self):
        self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=10000, parametric=False,
            random_state=self.seed
            )

    def time_bs_samples_parametric_500(self):
        self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=500, parametric=True,
            random_state=self.seed
            )

    def time_bs_samples_parametric_1000(self):
        self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=1000, parametric=True,
            random_state=self.seed
            )

    def time_bs_samples_parametric_2500(self):
        self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=2500, parametric=True,
            random_state=self.seed
            )   

    def time_bs_samples_parametric_5000(self):
        self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=5000, parametric=True,
            random_state=self.seed
            )

    def time_bs_samples_parametric_10000(self):
        self.bcl._bs_samples(
            self.sampling_dist, self.tri_fit_incr, sims=10000, parametric=True,
            random_state=self.seed
            )

    def time_bs_ldfs(self):
        self.bcl._bs_ldfs(self.dfsamples)

    def time_bs_forecasts(self):
        self.bcl._bs_forecasts(self.dfsamples, self.dfldfs, self.scale_param)

    def time_bs_process_error(self):
        self.bcl._bs_process_error(self.dfsamples, self.dfldfs, self.scale_param)

    def time_bs_reserves(self):
        self.bcl._bs_reserves(self.dfprocess_error)

    def time_bclr_origin_dist(self):
        self.bclr.origin_dist

    def time_bclr_agg_dist(self):
        self.bclr.agg_dist

    def time_bclr_residuals_detail(self):
        self.bclr.residulas_detail

    def time_bclr_bs_data_transform(self):
        self.bclr._bs_data_transform(self.qtls, self.qtlhdrs)

    def time_bclr_get_quantiles_by_devp(self):
        self.bclr._get_quantiles_by_devp(self.qtls, self.qtlhdrs)

    def time_bcl_500(self):
        self.bcl.__call__(sims=500, parametric=False, random_state=self.seed)

    def time_bcl_1000(self):
        self.bcl.__call__(sims=1000, parametric=False, random_state=self.seed)

    def time_bcl_2500(self):
        self.bcl.__call__(sims=2500, parametric=False, random_state=self.seed)

    def time_bcl_5000(self):
        self.bcl.__call__(sims=5000, parametric=False, random_state=self.seed)

    def time_bcl_10000(self):
        self.bcl.__call__(sims=10000, parametric=False, random_state=self.seed)

    def time_bcl_25000(self):
        self.bcl.__call__(sims=25000, parametric=False, random_state=self.seed)

    def time_bcl_parametric_500(self):
        self.bcl.__call__(sims=500, parametric=True, random_state=self.seed)

    def time_bcl_parametric_1000(self):
        self.bcl.__call__(sims=1000, parametric=True, random_state=self.seed)

    def time_bcl_parametric_2500(self):
        self.bcl.__call__(sims=2500, parametric=True, random_state=self.seed)

    def time_bcl_parametric_5000(self):
        self.bcl.__call__(sims=5000, parametric=True, random_state=self.seed)

    def time_bcl_parametric_10000(self):
        self.bcl.__call__(sims=10000, parametric=True, random_state=self.seed)

    def time_bcl_parametric_25000(self):
        self.bcl.__call__(sims=25000, parametric=True, random_state=self.seed)



class MackChainLadderSuite:
    def setup(self):
        data = trikit.load(dataset="raa")
        self.tri = trikit.totri(data=data, type_="cumulative")
        self.mcl = chainLadder.MackChainLadder(self.tri)
        self.ldfs = self.mcl._ldfs(alpha=1)
        self.devpvar = self._devp_variance(self.ldfs, alpha=1)
        self.ldfvar = self.mcl._ldf_variance(self.devpvar, alpha=1)
        self.mclr = self.mcl.__call__(alpha=1, tail=1, dist="lognorm")

    def time_mcl_mod_tri(self):
        self.mcl.mod_tri

    def time_mcl_mod_a2aind(self):
        self.mcl.mod_a2aind

    def time_mcl_ldfs_alpha_0(self):
        self.mcl._ldfs(alpha=0)

    def time_mcl_ldfs_alpha_1(self):
        self.mcl._ldfs(alpha=1)

    def time_mcl_ldfs_alpha_2(self):
        self.mcl._ldfs(alpha=2)

    def time_mcl_ldf_variance(self):
        self.mcl._ldf_variance(alpha=1)

    def time_mcl_devp_variance(self):
        self.mcl._devp_variance(alpha=1)

    def time_mcl_process_error(self):
        self.mcl._process_error(self.ldfs, self.devpvar)

    def time_mcl_parameter_error(self):
        self.mcl._parameter_error(self.ldfs, self.ldfvar)
