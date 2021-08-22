
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





Sample datasets are accessed using trikit's ``load`` function. Pass any of the sample dataset names
listed above as a string. The result will be a Pandas DataFrame of incremental losses. For
example, to load the **awz09** dataset, run::

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


Available sample datasets can be listed by calling ``get_datasets``::

    In [1]: from trikit import get_datasets
    In [2]: get_datasets()
    Out[1]:
    ['raa',
     'ta83',
     'lrdb',
     'amw09',
     'autoliab',
     'glre',
     'singinjury',
     'singproperty']
