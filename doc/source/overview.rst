
.. _overview:

================================================================================
Project Overview
================================================================================

trikit is a collection of Actuarial Reserving tools and techniques written in 
Python. The library is built around triangle objects, which are data structures 
that organize losses by year of origin as a function of development period. 
trikit's `IncrTriangle` and `CumTriangle` classes inherit from Pandas DataFrame, 
so triangle objects can be manipulated using the familiar Pandas API.             


trikit is distributed with a number of sample datasets, in addition to the 
Commercial Auto dataset from the |LRDB|__. Example working with included 
sample datasets can be found in the :ref:`Quickstart Guide<quickstart>`. 


Installation
********************************************************************************

trikit can be installed by running::

	$ python -m pip install trikit
    

.. |LRDB| replace:: CAS Loss Reserving Database
__ https://www.casact.org/research/index.cfm?fa=loss_reserves_data