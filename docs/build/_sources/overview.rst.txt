
.. _overview:

================================================================================
Project Overview
================================================================================

trikit is a collection of Actuarial reserving utilities written in
Python. The library is built around triangle objects, which are data structures 
that organize losses by year of origin as a function of development period. 
trikit's IncrTriangle and CumTriangle classes inherit from Pandas DataFrame,
so triangle objects can be manipulated using the familiar Pandas API.             

trikit is distributed with a number of sample datasets, in addition to the |LRDB|__.
Examples working with included sample datasets can be found in the :ref:`Quickstart Guide<quickstart>`,
:ref:`datasets reference<datasets>` and the |LRDB| :ref:`reference <lrdb>`.



.. |LRDB| replace:: CAS Loss Reserving Database
__ https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p
