
.. _quickstart:

===============================================================================
Quickstart Guide
===============================================================================

This section attempts to get the user up and running with trikit
as quickly as possible. We first demonstrate how to create Triangle objects, 
then follow with a demonstration of the Chain Ladder method and how to
produce estimates of reserve variability.



Triangle Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The primary means of converting datasets to Triangle objects is with the 
``totri`` function. It is assumed that any dataset passed to ``totri`` 
has already been converted to a pandas DataFrame. In the examples that
follow, we'll demonstrate how to produce both incremental and cumulative
triangle data structures from different dataset starting points. 






Chain Ladder 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


In addition to Chain Ladder point estimates, trikit exposes methods to 
help quantify reserve variability. The ``chnladder`` function can accept
an optional ``range_method`` parameter, which can currently be set to 
``None`` (point estimate only), ``"mack"`` or ``"bootstrap"``. 



.. note:: The "mcmc" option for ``range_method``, based on Glenn Meyer's Correlated
	Chain Ladder formulation, is expected to be available in the next release. 
	Please check back periodically. 
	




Sample Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to stand-alone datasets, trikit is distributed along with the 
|LRDB|__. More information on accessing and working with the database can be 
found :ref:`here<lrdb>`.











.. |LRDB| replace:: CAS Loss Reserving Database
__ https://www.casact.org/research/index.cfm?fa=loss_reserves_data


