import sys; sys.path.append("G:\\Repos")
import unittest
import pandas as pd
import numpy as np
import os
import os.path
import decimal
import logging
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import profile
import cProfile
import io
import pstats
import trikit
from numpy.random import RandomState


TRIANGLE_TEST = False
DATASETS_TEST = False
BASE_CL_TEST  = False
BOOT_CL_TEST  = False
MACK_CL_TEST  = False
MCMC_CL_TEST  = False


triDF0 = pd.DataFrame({
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

df = pd.DataFrame({
    "group" : ["a", "a", "b", "b", "c"],
    "gender": ["M", "F", "F", "M", "F"],
    "score" : np.random.rand(5),
    "salary": [90000, 70000, 74000, 89000, 77000]
    })
df1 = pd.get_dummies(df, columns=["group", "gender"], prefix_sep="_", drop_first=True)
triDF1 = pd.DataFrame({
    "1":[1065,1042,1334,1177,1732],
    "2":[1493,1332,1887,1465,np.NaN],
    "3":[1201,1129,1465,np.NaN,np.NaN],
    "4":[966,1074,np.NaN,np.NaN,np.NaN],
    "5":[0,np.NaN,np.NaN,np.NaN,np.NaN]
    }, index=range(2014,2019)
    )


# Test load function =========================================================]
if DATASETS_TEST:

    lrdb = pd.read_csv("G:/Repos/trikit/datasets/lrdb.csv", sep=",")
    raa  = trikit.load(dataset="raa")
    ta83 = trikit.load(dataset="ta83")
    datasets = trikit.get_datasets()


    # Test calling _get_lrdb_groups with list, dict and pd.DataFrame.
    grps0 = trikit.get_lrdb_groups(returnas=pd.DataFrame)
    grps1 = trikit.get_lrdb_groups(returnas=dict)
    grps2 = trikit.get_lrdb_groups(returnas=list)
    lobs = trikit.get_lrdb_lobs()
    specs = trikit.get_lrdb_specs()


    dat1 = trikit.load(dataset="lrdb", grcode=2143, action=None)
    dat2 = trikit.load(dataset="lrdb", grcode=2143, action="random")
    dat3 = trikit.load(dataset="lrdb", grcode=2143, action="aggregate")

    # Call each lob with varying `action` values.
    dat4a = trikit.load(dataset="lrdb", lob="WC", action=None)
    dat4b = trikit.load(dataset="lrdb", lob="WC", action="agg")
    dat4c = trikit.load(dataset="lrdb", lob="WC", action="rand")

    # Return single group with only `action` specified.
    dat5a = trikit.load(dataset="lrdb", action="random")
    dat5b = trikit.load(dataset="lrdb", lower_right_ind=True, action="random")

    # Varying loss_type parameter.
    dat6a = trikit.load(dataset="lrdb", grcode=2003, loss_type="incurred", action="random")
    dat6b = trikit.load(dataset="lrdb", grcode=3240, loss_type="paid", action="random")
    dat6c = trikit.load(dataset="lrdb", grcode=715, loss_type="bulk", action="random")

    # Filter to full lob while varying `action`.
    # lob1a = trikit.load(dataset="lrdb", lob="WC"       , action="agg")
    # lob2a = trikit.load(dataset="lrdb", lob="PROD_LIAB", action="agg")
    # lob3a = trikit.load(dataset="lrdb", lob="PP_AUTO"  , action="agg")
    # lob4a = trikit.load(dataset="lrdb", lob="OTHR_LIAB", action="agg")
    # lob5a = trikit.load(dataset="lrdb", lob="MED_MAL"  , action="agg")
    # lob6a = trikit.load(dataset="lrdb", lob="COM_AUTO" , action="agg")
    # lob1b = trikit.load(dataset="lrdb", lob="WC"       , action="random")
    # lob2b = trikit.load(dataset="lrdb", lob="PROD_LIAB", action="random")
    # lob3b = trikit.load(dataset="lrdb", lob="PP_AUTO"  , action="random")
    # lob4b = trikit.load(dataset="lrdb", lob="OTHR_LIAB", action="random")
    # lob5b = trikit.load(dataset="lrdb", lob="MED_MAL"  , action="random")
    # lob6b = trikit.load(dataset="lrdb", lob="COM_AUTO" , action="random")


    ld0 = trikit.load(dataset="lrdb", loss_type="paid", action="aggregate")
    ld1 = trikit.load(dataset="lrdb", loss_type="paid", action=None)
    ld2 = trikit.load(dataset="lrdb", loss_type="paid", action="random")
    ld4 = trikit.load(dataset="lrdb", loss_type="paid", action="rand", allcols=False)


# Verify other parameters are ignored when dataset!="lrdb"




# totri testing ==============================================================]
# totri(data, type_="cumulative", origin=None, dev=None, value=None,
#       trifmt=None, datafmt="incremental", **kwargs)
# ============================================================================]
raa  = trikit.load(dataset="raa")
DATA = raa


# ti = trikit.triangle._IncrTriangle(data=DATA)
# tc = trikit.triangle._CumTriangle(data=DATA)
# tc.plot(facets=True)



if TRIANGLE_TEST:

    ##### Case 0 ##############################################################
    # data   :raa
    # type_  :incr
    # origin :None
    # dev    :None
    # value  :None
    # trifmt :None
    # datafmt:incr
    tri0 = trikit.totri(
        data=DATA, type_="incremental", origin=None, dev=None, value=None,
        datafmt="incremental"
        )

    rlvi    = tri0.rlvi
    clvi    = tri0.clvi
    latest  = tri0.latest
    lbo     = tri0.latest_by_origin
    lbd     = tri0.latest_by_devp
    devp    = tri0.devp
    origins = tri0.origins


    ##### Case 1 ##############################################################
    # data   :raa
    # type_  :"cum"
    # origin :None
    # dev    :None
    # value  :None
    # trifmt :None
    # datafmt:"incr"
    tri1 = trikit.totri(
        data=DATA, type_="cumulative", origin=None, dev=None, value=None,
        datafmt="incremental"
        )

    rlvi    = tri1.rlvi
    clvi    = tri1.clvi
    latest  = tri1.latest
    lbo     = tri1.latest_by_origin
    lbd     = tri1.latest_by_devp
    devp    = tri1.devp
    origins = tri1.origins
    linkr   = tri1.a2a
    linkra1  = tri1.a2a_avgs
    linkra2  = tri1.a2a_avgs
    matur   = tri1.maturity

    ##### Case 2 ##############################################################
    # data   :raacum
    # type_  :cum
    # origin :None
    # dev    :None
    # value  :None
    # trifmt :None
    # datafmt:cum
    # tri2 = tri1.cumsum(axis=1)

    raacum = trikit._tritotbl(tri1)
    tri2 = trikit.totri(data=raacum, type_="cumulative", datafmt="cumulative")

    rlvi    = tri2.rlvi
    clvi    = tri2.clvi
    latest  = tri2.latest
    lbo     = tri2.latest_by_origin
    lbd     = tri2.latest_by_devp
    devp    = tri2.devp
    origins = tri2.origins
    linkr   = tri2.a2a
    linkra  = tri2.a2a_avgs
    matur   = tri1.maturity


    ##### Case 3 ##############################################################
    # data   :raacum
    # type_  :cum
    # origin :None
    # dev    :None
    # value  :None
    # trifmt :None
    # datafmt:incr
    raacum = tri1.as_tbl()
    tri3 = trikit.totri(data=raacum, type_="incremental", datafmt="cumulative")

    rlvi    = tri3.rlvi
    clvi    = tri3.clvi
    latest  = tri3.latest
    lbo     = tri3.latest_by_origin
    lbd     = tri3.latest_by_devp
    devp    = tri3.devp
    origins = tri3.origins
    matur   = tri1.maturity


    ##### Case 4 ##############################################################
    # data   :raa
    # type_  :cum
    # origin :None
    # dev    :None
    # value  :None
    # trifmt :None
    # datafmt:""
    cumtri = pd.DataFrame(tri1)
    tri4 = trikit.totri(data=cumtri, type_="cumulative", trifmt="cumulative")

    rlvi    = tri4.rlvi
    clvi    = tri4.clvi
    latest  = tri4.latest
    lbo     = tri4.latest_by_origin
    lbd     = tri4.latest_by_devp
    devp    = tri4.devp
    origins = tri4.origins
    linkr   = tri4.a2a
    linkra  = tri4.a2a_avgs
    matur   = tri1.maturity


    ##### Case 5 ##############################################################
    # data   :raa
    # type_  :cum
    # origin :None
    # dev    :None
    # value  :None
    # trifmt :None
    # datafmt:""
    cumtri = pd.DataFrame(tri1)
    tri5 = trikit.totri(
        data=cumtri, type_="incremental", trifmt="cumulative"
        )

    rlvi    = tri5.rlvi
    clvi    = tri5.clvi
    latest  = tri5.latest
    lbo     = tri5.latest_by_origin
    lbd     = tri5.latest_by_devp
    devp    = tri5.devp
    origins = tri5.origins
    matur   = tri1.maturity


    ##### Case 6 ##############################################################
    # data   :raa
    # type_  :incr
    # origin :None
    # dev    :None
    # value  :None
    # trifmt :None
    # datafmt:""
    incrtri = pd.DataFrame(tri0)
    tri6 = trikit.totri(
        data=incrtri, type_="incremental", trifmt="incremental"
        )

    rlvi    = tri6.rlvi
    clvi    = tri6.clvi
    latest  = tri6.latest
    lbo     = tri6.latest_by_origin
    lbd     = tri6.latest_by_devp
    devp    = tri6.devp
    origins = tri6.origins
    matur   = tri1.maturity


    ##### Case 7 ##############################################################
    # data   :raa
    # type_  :incr
    # origin :None
    # dev    :None
    # value  :None
    # trifmt :None
    # datafmt:""
    incrtri = pd.DataFrame(tri0)
    tri7 = trikit.totri(
        data=incrtri, type_="cumulative", trifmt="incremental"
        )

    rlvi    = tri7.rlvi
    clvi    = tri7.clvi
    latest  = tri7.latest
    lbo     = tri7.latest_by_origin
    lbd     = tri7.latest_by_devp
    devp    = tri7.devp
    origins = tri7.origins
    linkr   = tri7.a2a
    linkra  = tri7.a2a_avgs
    matur   = tri1.maturity

#
# cldfs0a = r.cldfs
# ults0a  = r.ultimates
# ibnr0a  = r.reserves
# tsqr0a  = r.trisqrd




# from trikit.chainladder import bootstrap
# bb = bootstrap._BootstrapChainLadder(cumtri=tri1)

#
# dof = bb.tri.dof
# tfc = bb._tri_fit_cum(sel="all-weighted")
# tfi = bb._tri_fit_incr(fitted_tri_cum=tfc)
# r_us = bb._resid_us(tfi)
# r_adj = bb._resid_adj(resid_us=r_us)
# sclp = bb._scale_param(r_us)
# sampling_dist = bb._sampling_dist(r_adj)
#
# dfsamples = bb._bs_samples(
#     sampling_dist=sampling_dist, fitted_tri_incr=tfi, sims=10, neg_handler=1,
#     parametric=False, random_state=516
#     )
#
# dfldfs = bb._bs_ldfs(dfsamples=dfsamples)
#
# rlvi_ = bb.tri.rlvi.reset_index(drop=False).rename({"index":"origin", "dev":"l_act_dev"}, axis=1)
# dflvi = rlvi_.drop("col_offset", axis=1)
# dfcombined = dfsamples.merge(dfldfs, on=["sim", "dev"], how="left")
# dfcombined = dfcombined.merge(dflvi, how="left", on=["origin"])
# dfcombined = dfcombined.reset_index(drop=True).sort_values(by=["sim", "origin", "dev"])
# dfforecasts = bb._bs_forecasts(dfcombined=dfcombined, scale_param=sclp)
# dfprocerr = bb._bs_process_error(dfforecasts=dfforecasts, scale_param=sclp, procdist="gamma", random_state=516)
# dfreserves = bb._bs_reserves(dfprocerror=dfprocerr)




# Testing _BaseChainLadder  ==================================================]
#
# def chladder(data, origin=None, dev=None, value=None, trifmt=None,
#              datafmt="incremental", tail=1.0, sel="all-weighted",
#              range_method=None, **kwargs)
#
# ============================================================================]

if BASE_CL_TEST:
    # Passing tabular dataset with fields "origin", "dev", "value"
    cl0     = trikit.chladder(data=DATA)
    ldfs0a  = cl0.ldfs
    cldfs0a = cl0.cldfs
    ults0a  = cl0.ultimates
    ibnr0a  = cl0.reserves
    tsqr0a  = cl0.trisqrd

    # Change ldf at `1-2` index; verify change flows through to other properties,
    # starting with self.cldfs > self.ultimates > self.reserves.
    cl0.ldfs = ("1-2", 2.15)

    # Attempt to update non-existent index (results in no-op).
    cl0.ldfs = ("15-16", 1.0034)
    ldfs0b   = cl0.ldfs
    cldfs0b  = cl0.cldfs
    ults0b   = cl0.ultimates
    ibnr0b   = cl0.reserves
    tsqr0b   = cl0.trisqrd

    assert not ldfs0a.equals(ldfs0b)
    assert not cldfs0a.equals(cldfs0b)
    assert not ults0a.equals(ults0b)
    assert not ibnr0a.equals(ibnr0b)
    assert not tsqr0a.equals(tsqr0b)


    # Passing data formatted as cumulative triangle but not typed as such.
    dfc = pd.DataFrame(tri1)
    dfi = trikit._cumtoincr(dfc)
    cl1 = trikit.chladder(data=dfc, trifmt="cumulative")
    cl2 = trikit.chladder(data=dfi, trifmt="incremental")
    assert cl0().equals(cl1)==cl1().equals(cl2)







# Testing _BootstrapChainLadder ==============================================]
# chladder(data, origin=None, dev=None, value=None, trifmt=None,
#              datafmt="incremental", tail=1.0, sel="all-weighted",
#              range_method=None, **kwargs):
if BOOT_CL_TEST:

    bcl0 = trikit.chladder(data=DATA, range_method="bootstrap",)


    NBR_SIMS    = 10
    prng        = RandomState(20160516)
    tri         = bcl0.tri
    latest      = bcl0.tri.latest_by_origin.sort_index()
    sel         = bcl0.sel
    tail        = bcl0.tail
    nh          = bcl0.neg_handler
    ldfs0       = bcl0.ldfs
    cldfs0      = bcl0.cldfs
    tsqr        = bcl0.trisqrd.round(0)
    ults0       = bcl0.ultimates
    ibnr0       = bcl0.reserves
    ncells      = bcl0.nbr_cells
    dof         = bcl0.dof
    sclpar      = bcl0.scale_param
    trifitc     = bcl0.tri_fit_cum
    trifiti     = bcl0.tri_fit_incr
    resid_us    = bcl0.resid_us
    resid_adj   = bcl0.resid_adj
    sdist       = bcl0.sampling_dist
    psamples    = bcl0._bs_samples(sims=NBR_SIMS, random_state=prng, parametric=True)
    npsamples   = bcl0._bs_samples(sims=NBR_SIMS, random_state=prng)
    pldfs       = bcl0._bs_ldfs(psamples)
    npldfs      = bcl0._bs_ldfs(npsamples)
    pforecasts  = bcl0._bs_forecasts(samples=psamples, ldfs=pldfs)
    npforecasts = bcl0._bs_forecasts(samples=npsamples, ldfs=npldfs)
    pperror     = bcl0._bs_process_error(pforecasts, procdist="gamma")
    npperror    = bcl0._bs_process_error(npforecasts, procdist="gamma")
    fit_assess  = bcl0.fit_assessment
    residsumm   = bcl0.residuals_detail


# Run bcl0.__call__ ==========================================================]

# __call__(sims=1000, procdist="gamma", parametric=False,
#          percentiles=[.75, .95], interpolation="linear",
#          summary=True, random_state=None)
#
# ============================================================================]

# _ldfs       = self._bs_ldfs(samples=_samples)
# _forecasts  = self._bs_forecasts(samples=_samples, ldfs=_ldfs)
# _proc_error = self._bs_process_error(forecasts=_forecasts, procdist=procdist)
# _reserves   = self._bs_reserves(_proc_error)






# Testing MackChainLadder ====================================================]
tri = trikit.triangle._CumTriangle(data=RAA)
cl = trikit._BaseChainLadder(tri).__call__()

cl = cl.__call__()


# if MACK_CL_TEST:
DATA = trikit.load("ta83")
tri = trikit.totri(DATA, type_="cumulative")
mcl = trikit._MackChainLadder(cumtri=tri)
ldfs_ = mcl._ldfs(alpha=1, tail=1.0)
devpvar_ = mcl._devpvar(alpha=1, tail=1.0)




if MCMC_CL_TEST:

    pass








# TESTING_STATE can be either "print" or "log" ===============================]
# TESTING_STATE = "print"
#
#
# if TESTING_STATE=="print:
#
#     # _IncrTriangle Tests =>
#     print("<>=<>=<>=<> `_IncrTriangle` => i1 [Actual Data] <>=<>=<>=<>")
#     print("i1.latest_diag     : {}".format(i1.latest_diag))
#     print("i1.dev_periods     : {}".format(i1.dev_periods))
#     print("i1.origin_yrs      : {}".format(i1.origin_yrs))
#     print("i1.get_origin(1989): {}".format(i1.get_origin(1989)))
#     print("i1.get_dev(4)      : {}".format(i1.get_dev(4)))
#     print("")
#
#     print("<>=<>=<>=<> `_IncrTriangle` => i2 [All NaNs]    <>=<>=<>=<>")
#     print("i2.latest_diag     : {}".format(i2.latest_diag))
#     print("i2.dev_periods     : {}".format(i2.dev_periods))
#     print("i2.origin_yrs      : {}".format(i2.origin_yrs))
#     print("i2.get_origin(1989): {}".format(i2.get_origin(1989)))
#     print("i2.get_dev(4)      : {}".format(i2.get_dev(4)))
#     print("")
#
#     print("<>=<>=<>=<> `_IncrTriangle` => i3 [0-rows/cols] <>=<>=<>=<>")
#     print("i3.latest_diag     : {}".format(i3.latest_diag))
#     print("i3.dev_periods     : {}".format(i3.dev_periods))
#     print("i3.origin_yrs      : {}".format(i3.origin_yrs))
#     print("i3.get_origin(1989): {}".format(i3.get_origin(1989)))
#     print("i3.get_dev(4)      : {}".format(i3.get_dev(4)))
#     print("")
#
#
#     # _CumTriangle Tests =>
#     print("<>=<>=<>=<> `_CumTriangle` - c1 - [Actual Data] <>=<>=<>=<>")
#     print("c1.latest_diag              : {}".format(c1.latest_diag))
#     print("c1.dev_periods              : {}".format(c1.dev_periods))
#     print("c1.origin_yrs               : {}".format(c1.origin_yrs))
#     print("c1.get_origin(1989)         : {}".format(c1.get_origin(1989)))
#     print("c1.get_dev(4)               : {}".format(c1.get_dev(4)))
#     print("c1.ldfs                      : {}".format(c1.ldfs))
#     print("c1.ldfs_avgs()               :\n{}\n".format(c1.ldfs_avgs()))
#     print("c1.ldfs_avgs(addl_avgs=8)    :\n{}\n".format(c1.ldfs_avgs(addl_avgs=8)))
#     print("c1.ldfs_avgs(addl_avgs=[6,8]):\n{}\n".format(c1.ldfs_avgs(addl_avgs=[6,8])))
#     print("")
#
#     print("<>=<>=<>=<> `_CumTriangle` - c2 - [All NaNs]    <>=<>=<>=<>")
#     print("c2.latest_diag              : {}".format(c2.latest_diag))
#     print("c2.dev_periods              : {}".format(c2.dev_periods))
#     print("c2.origin_yrs               : {}".format(c2.origin_yrs))
#     print("c2.get_origin(1989)         : {}".format(c2.get_origin(1989)))
#     print("c2.get_dev(4)               : {}".format(c2.get_dev(4)))
#     print("c2.ldfs                      : {}".format(c2.ldfs))
#     print("c2.ldfs_avgs()               :\n{}\n".format(c2.ldfs_avgs()))
#     print("c2.ldfs_avgs(addl_avgs=6)    :\n{}\n".format(c2.ldfs_avgs(addl_avgs=6)))
#     print("c2.ldfs_avgs(addl_avgs=[4,9]):\n{}\n".format(c2.ldfs_avgs(addl_avgs=[4,9])))
#     print("")
#
#     print("<>=<>=<>=<> `_CumTriangle` - c3 - [0-rows/cols] <>=<>=<>=<>")
#     print("c3.latest_diag              : {}".format(c3.latest_diag))
#     print("c3.dev_periods              : {}".format(c3.dev_periods))
#     print("c3.origin_yrs               : {}".format(c3.origin_yrs))
#     print("c3.get_origin(1989)         : {}".format(c3.get_origin(1989)))
#     print("c3.get_dev(4)               : {}".format(c3.get_dev(4)))
#     print("c3.ldfs                      : {}".format(c3.ldfs))
#     print("c3.ldfs_avgs()               :\n{}\n".format(c3.ldfs_avgs()))
#     print("c3.ldfs_avgs(addl_avgs=4)    :\n{}\n".format(c3.ldfs_avgs(addl_avgs=4)))
#     print("c3.ldfs_avgs(addl_avgs=[6,4]):\n{}\n".format(c3.ldfs_avgs(addl_avgs=[6,4])))
#     print("")
#
#
#
#
# elif TESTING)STATE=="log":
#
#     logging.basicConfig(
#             filename="U:/trikit/Logging/trikit_unittests.txt",
#             level=logging.DEBUG,
#             format='%(asctime)s - %(levelname)s - %(message)s'
#             )
#
#     # _IncrTriangle Tests =>
#     logging.debug("<>=<>=<>=<> `_IncrTriangle` => i1 [Actual Data] <>=<>=<>=<>")
#     logging.debug("i1.latest_diag     : {}".format(i1.latest_diag))
#     logging.debug("i1.dev_periods     : {}".format(i1.dev_periods))
#     logging.debug("i1.origin_yrs      : {}".format(i1.origin_yrs))
#     logging.debug("i1.get_origin(1989): {}".format(i1.get_origin(1989)))
#     logging.debug("i1.get_dev(4)      : {}".format(i1.get_dev(4)))
#     logging.debug("")
#
#     logging.debug("<>=<>=<>=<> `_IncrTriangle` => i2 [All NaNs]    <>=<>=<>=<>")
#     logging.debug("i2.latest_diag     : {}".format(i2.latest_diag))
#     logging.debug("i2.dev_periods     : {}".format(i2.dev_periods))
#     logging.debug("i2.origin_yrs      : {}".format(i2.origin_yrs))
#     logging.debug("i2.get_origin(1989): {}".format(i2.get_origin(1989)))
#     logging.debug("i2.get_dev(4)      : {}".format(i2.get_dev(4)))
#     logging.debug("")
#
#     logging.debug("<>=<>=<>=<> `_IncrTriangle` => i3 [0-rows/cols] <>=<>=<>=<>")
#     logging.debug("i3.latest_diag     : {}".format(i3.latest_diag))
#     logging.debug("i3.dev_periods     : {}".format(i3.dev_periods))
#     logging.debug("i3.origin_yrs      : {}".format(i3.origin_yrs))
#     logging.debug("i3.get_origin(1989): {}".format(i3.get_origin(1989)))
#     logging.debug("i3.get_dev(4)      : {}".format(i3.get_dev(4)))
#     logging.debug("")
#
#
#
#     # _CumTriangle Tests =>
#     logging.debug("<>=<>=<>=<> `_CumTriangle` - c1 - [Actual Data] <>=<>=<>=<>")
#     logging.debug("c1.latest_diag              : {}".format(c1.latest_diag))
#     logging.debug("c1.dev_periods              : {}".format(c1.dev_periods))
#     logging.debug("c1.origin_yrs               : {}".format(c1.origin_yrs))
#     logging.debug("c1.get_origin(1989)         : {}".format(c1.get_origin(1989)))
#     logging.debug("c1.get_dev(4)               : {}".format(c1.get_dev(4)))
#     logging.debug("c1.ldfs                      : {}".format(c1.ldfs))
#     logging.debug("c1.ldfs_avgs()               :\n{}\n".format(c1.ldfs_avgs()))
#     logging.debug("c1.ldfs_avgs(addl_avgs=8)    :\n{}\n".format(c1.ldfs_avgs(addl_avgs=8)))
#     logging.debug("c1.ldfs_avgs(addl_avgs=[6,8]):\n{}\n".format(c1.ldfs_avgs(addl_avgs=[6,8])))
#     logging.debug("")
#
#     logging.debug("<>=<>=<>=<> `_CumTriangle` - c2 - [All NaNs]    <>=<>=<>=<>")
#     logging.debug("c2.latest_diag              : {}".format(c2.latest_diag))
#     logging.debug("c2.dev_periods              : {}".format(c2.dev_periods))
#     logging.debug("c2.origin_yrs               : {}".format(c2.origin_yrs))
#     logging.debug("c2.get_origin(1989)         : {}".format(c2.get_origin(1989)))
#     logging.debug("c2.get_dev(4)               : {}".format(c2.get_dev(4)))
#     logging.debug("c2.ldfs                      : {}".format(c2.ldfs))
#     logging.debug("c2.ldfs_avgs()               :\n{}\n".format(c2.ldfs_avgs()))
#     logging.debug("c2.ldfs_avgs(addl_avgs=6)    :\n{}\n".format(c2.ldfs_avgs(addl_avgs=6)))
#     logging.debug("c2.ldfs_avgs(addl_avgs=[4,9]):\n{}\n".format(c2.ldfs_avgs(addl_avgs=[4,9])))
#     logging.debug("")
#
#     logging.debug("<>=<>=<>=<> `_CumTriangle` - c3 - [0-rows/cols] <>=<>=<>=<>")
#     logging.debug("c3.latest_diag              : {}".format(c3.latest_diag))
#     logging.debug("c3.dev_periods              : {}".format(c3.dev_periods))
#     logging.debug("c3.origin_yrs               : {}".format(c3.origin_yrs))
#     logging.debug("c3.get_origin(1989)         : {}".format(c3.get_origin(1989)))
#     logging.debug("c3.get_dev(4)               : {}".format(c3.get_dev(4)))
#     logging.debug("c3.ldfs                      : {}".format(c3.ldfs))
#     logging.debug("c3.ldfs_avgs()               :\n{}\n".format(c3.ldfs_avgs()))
#     logging.debug("c3.ldfs_avgs(addl_avgs=4)    :\n{}\n".format(c3.ldfs_avgs(addl_avgs=4)))
#     logging.debug("c3.ldfs_avgs(addl_avgs=[6,4]):\n{}\n".format(c3.ldfs_avgs(addl_avgs=[6,4])))
#     logging.debug("")
