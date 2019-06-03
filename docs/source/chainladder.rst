
.. _chainladder:

===============================================================================
The Chain Ladder Method
===============================================================================

This is the Chain Ladder page.




tri1.a2a.mask(tri1.a2a > 3)
tri1.a2a.where(tri1.a2a < 5.0) # Get link ratios less than 5, otherwise NaN.

# Remove link ratios greater than 3:
tri1._a2aind = tri1.a2a.mask(tri1.a2a > 3).applymap(lambda v: 0 if np.isnan(v) else 1)

.. autoclass:: trikit.chainladder._BaseChainLadder
	:members:

.. autoclass:: trikit.chainladder.mack._MackChainLadder
	:members:

.. autoclass:: trikit.chainladder.bootstrap._BootstrapChainLadder
	:members:

