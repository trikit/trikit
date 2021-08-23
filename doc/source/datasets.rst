
.. _datasets:

===============================================================================
Sample Datasets
===============================================================================

.. _rmafa: https://instruction.bus.wisc.edu/jfrees/jfreesbooks/Regression%20Modeling/BookWebDec2010/data.html

.. note::

    For information on accessing the CAS Loss Reserving Database from trikit, see `here <lrdb>`_.




trikit includes a number a sample loss datasets. As of the latest release,
the following are available:


**autoliab**
    Private passenger auto liability/medical coverages from 2004.
    Source: Frees, E., `Regression Modeling with Actuarial and Financial Applications <rmafa>`_.


**amw09**
    Sample loss dataset.
    Source: Alai, D.H. et. al., *Mean Square Error of Predicition in the Bornhuetter-Ferguson Claims Reserving Method*,
    Annals of Actuarial Science **4**, I, 7-31 (2009).


**glre**
    Reinsurance General Liability dataset from the 2001 *Historical Loss Development Study* published by the Reinsurance
    Association of America.
    Source: Frees, E., `Regression Modeling with Actuarial and Financial Applications <rmafa>`_.


**raa**
    Automatic Factultative business in General Liability provided by the Reinsurance Association of America.
    Source: Mack, Thomas (1993) *Measuring the Variability of Chain Ladder Reserve Estimates*, 1993 CAS Prize Paper Competition on
    Variability of Loss Reserves.

**singinjury**
    Payments from a portfolio of automobile policies for a Singapore property and casualty insurer.
    Source: Source: Frees, E., `Regression Modeling with Actuarial and Financial Applications <rmafa>`_.


**singproperty**
    Incremental payments from a portfolio of automobile policies for a Singapore property and casualty insurer.
    Source: Source: Frees, E., `Regression Modeling with Actuarial and Financial Applications <rmafa>`_.


**ta83**
    Sample loss dataset.
    Source: G. C. Taylor and F. R. Ashe, *Second moments of estimates of outstanding claims*, Journal of Econometrics, 1983, vol. 23, issue 1, 37-61.








Sample datasets are accessed using trikit's ``load`` function:


.. function:: load(dataset, tri_type=None)

   	Load the specified sample dataset. If ``tri_type`` is not None, return sample
	dataset as specified triangle either "cum" or "incr".

    :param dataset: Specifies which sample dataset to load. The complete set of sample
    datasets can be obtained by calling ``get_datasets``.
    :type dataset: str

	:param tri_type: If ``None``, lrdb subset is returned as pd.DataFrame. Otherwise,
    return subset as either incremental or cumulative triangle type. Default value is None.
    :type tri_type: {None, "incr", "cum"}

    :return: Either pd.DataFrame, trikit.triangle.IncrTriangle or trikit.triangle.CumTriangle.





Pass any of the sample dataset names. For example, to load the **amw09** dataset as a DataFrame, run::

    In [1]: from trikit import load
    In [2]: df = load("amw09")
    In [3]: df.head()
    Out[3]:
       origin  dev      value
    0       0    0  5946975.0
    1       0    1  3721237.0
    2       0    2   895717.0
    3       0    3   207760.0
    4       0    4   206704.0


To return the **awz09** sample dataset as a triangle of incremental losses, run::

    In [4]: tri = load("amw09", tri_type="incr")
    In [5]: tri
    Out[5]:
              0         1       2       3       4       5      6      7      8      9
    0 5,946,975 3,721,237 895,717 207,760 206,704  62,124 65,813 14,850 11,130 15,813
    1 6,346,756 3,246,406 723,222 151,797  67,824  36,603 52,752 11,186 11,646    nan
    2 6,269,090 2,976,233 847,053 262,768 152,703  65,444 53,545  8,924    nan    nan
    3 5,863,015 2,683,224 722,532 190,653 132,976  88,340 43,329    nan    nan    nan
    4 5,778,885 2,745,229 653,894 273,395 230,288 105,224    nan    nan    nan    nan
    5 6,184,793 2,828,338 572,765 244,899 104,957     nan    nan    nan    nan    nan
    6 5,600,184 2,893,207 563,114 225,517     nan     nan    nan    nan    nan    nan
    7 5,288,066 2,440,103 528,043     nan     nan     nan    nan    nan    nan    nan
    8 5,290,793 2,357,936     nan     nan     nan     nan    nan    nan    nan    nan
    9 5,675,568       nan     nan     nan     nan     nan    nan    nan    nan    nan


Available sample datasets can be listed by calling ``get_datasets``::

    In [6]: from trikit import get_datasets
    In [7]: get_datasets()
    Out[7]: ['amw09', 'autoliab', 'glre', 'raa', 'singinjury', 'singproperty', 'ta83']
