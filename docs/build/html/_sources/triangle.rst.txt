
.. _triangle:


:mod:`triangle` --- Incremental and Cumulative Triangle Objects
===============================================================================


.. note:: This page describes the ``triangle`` module in detail. For examples
	of practical use cases, please refer to the :ref:`Quickstart Guide<quickstart>`.
    
    
The loss triangle serves as the starting point for many reserving analyses.
Data is typically aggregated by year of origin and development period,
with origin year increasing top to bottom, and development period increasing
left to right. 





.. autoclass:: trikit.triangle._BaseTriangle
	:members:


.. autoclass:: trikit.triangle.incremental.IncrTriangle
	:members:


.. autoclass:: trikit.triangle.cumulative.CumTriangle
	:members:





         
The ``triangle`` module exposes three classes: ``_BaseTriangle``,
``IncrTriangle`` and ``CumTriangle``. ``_BaseTriangle`` inherits from 
``pandas.DataFrame``, which then serves as the base class for ``IncrTriangle`` 
and ``CumTriangle`` classes.
For the vast majority of use cases, neither ``IncrTriangle`` nor 
``CumTriangle`` should be invoked directly. Instead, the totri_ function 
is the preferred approach for initializing Triangle objects. 






serves as a base class 
for ``IncrTriangle`` and ``CumTriangle``, and should not be invoked directly.
For the vast majority of use cases, ``IncrTriangle`` or
``CumTriangle`` will not be instantiated directly. Instead, the ``totri``
function is the preferred method to convert datasets to Triangle objects. 


class constructor

``IncrTriangle`` and ``CumTriangle`` w




A typical example of an incremental loss triangle looks like:


+--------+------+------+------+------+------+
| ORIGIN | 1    | 2    | 3    | 4    | 5    |
+========+======+======+======+======+======+
| 2014   | 1065 | 1493 | 1201 | 966  | 0    |
+--------+------+------+------+------+------+
| 2015   | 1042 | 1332 | 1129 | 1074 |      |
+--------+------+------+------+------+------+
| 2016   | 1334 | 1887 | 1465 |      |      |
+--------+------+------+------+------+------+
| 2017   | 1177 | 1673 |      |      |      |
+--------+------+------+------+------+------+
| 2018   | 1732 |      |      |      |      |
+--------+------+------+------+------+------+

           


.. csv-table:: Incremental Loss Triangle
    :header: "ORIGIN", "1", "2", "3", "4", "5"

    2014, 1065, 1493, 1201, 966, 0
    2015, 1042, 1332, 1129, 1074,
    2016, 1334, 1887, 1465,,
    2017, 1177, 1673,,,
    2018, 1732,,,,

      
    
The cell containing the value "1332" represents losses paid (or incurred) 
between development periods **1** and **2** (between 12-24 months if each 
development period represents one year) in origin year **2015**.         


A Cumulative Triangle instance is compiled using the same underlying data, but 
summarizes cumulative loss amounts across development periods for each origin 
period. The cumulative version of the incremental triangle above is shown below.    



+--------+------+------+------+------+------+
| ORIGIN | 1    | 2    | 3    | 4    | 5    |
+========+======+======+======+======+======+
| 2014   | 1065 | 2558 | 3759 | 4725 | 4725 |
+--------+------+------+------+------+------+
| 2015   | 1042 | 2374 | 3503 | 4577 |      |
+--------+------+------+------+------+------+
| 2016   | 1334 | 3221 | 4686 |      |      |
+--------+------+------+------+------+------+
| 2017   | 1177 | 2850 |      |      |      |
+--------+------+------+------+------+------+
| 2018   | 1732 |      |      |      |      |
+--------+------+------+------+------+------+

|

.. csv-table:: Cumulative Loss Triangle
    :header: "ORIGIN", "1", "2", "3", "4", "5"

    2014, 1065, 2558, 3759, 4725, 4725
    2015, 1042, 2374, 3503, 4577,
    2016, 1334, 3221, 4686,,
    2017, 1177, 2850,,,
    2018, 1732,,,,


     

In the cumulative triangle, the value "4686" represents losses paid (or incurred) 
to date for origin year 2016, which includes development periods 1-3 (between
0-36 months if each development period represents one year).
