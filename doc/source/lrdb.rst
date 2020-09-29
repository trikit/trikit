
.. _lrdb:

===============================================================================
The CAS Loss Reserving Database
===============================================================================

----------------------------------------------------

The |LRDB|__ is a collection of loss data originally compiled by Glenn Meyers 
intended to be used for claims reserving studies. The data includes major 
personal and commercial lines of business from U.S. property casualty insurers.
The claims data comes from 
*Schedule P Analysis of Losses and Loss Expenses* in the National
Association of Insurance Commissioners (NAIC) database. The final product is a 
dataset that contains run-off triangles of six lines of business for all U.S. 
property casualty insurers. The triangle data correspond to claims of accident 
year 1988-1997 with 10 years development lag. Both upper and lower triangles
are included so that one could use the data to develop a model and then test 
its performance retrospectively [#f1]_.   

     
| The |LRDB| contains the following fields:


``loss_key``            
	One of "WC", "PROD_LIAB", "PP_AUTO", "OTHR_LIAB", "MED_MAL" or "COM_AUTO"
	(see mapping below)

``loss_type_suffix``           
	One of "D", "R", "B", "H", "F" or "C" (see mapping below)                 

``grcode``       
	NAIC company code (including insurer groups and single insurers)               

``grname``      	    
	NAIC company name (including insurer groups and single insurers)           
  
``origin``        
	Accident year          

``dev``	   
	Development year                 

``incrd_loss``   
	Incurred losses and allocated expenses reported at year end       

``paid_loss``    
	paid losses and allocated expenses at year end       

``bulk_loss``            
	Bulk and IBNR reserves on net losses and defense and cost containment 
	expenses reported at year end

``earned_prem_dir``   	
	Premiums earned at incurral year - direct and assumed     

``earned_prem_ceded``       
	Premiums earned at incurral year - ceded       
	
``earned_prem_net``       	
	earned_prem_dir - earned_prem_ceded      

``single``         	
	**1** indicates a single entity, **0** indicates a group insurer         

``posted_reserve_97``     
	Posted reserves in year 1997 taken from the Underwriting and Investment 
	Exhibit Part 2A, including net losses unpaid and unpaid loss adjustment
	expenses     

``upper_left_ind``	      
	**1** indicates whether the value associated with a particular 
	``origin-dev`` combination would fall in the upper-left of a typical loss 
	triangle   

``lower_right_ind``
	**1** corresponds to values that are typically calculated using a reserve 
	estimation technique, such as the chain ladder method. However, within the 
	context of the |LRDB|, there are 10 years of development lag for each 
	origin year. This field serves as a mechanism to exclude these additional 
	development periods for analyses requiring a more conventional loss 
	triangle structure, where the latest origin year has a one period of 
	development, the second latest origin year has two periods of development, 
	etc.


The following table provides a description of the type of losses associated 
with each unique combination of ``loss_key`` and
``loss_type_suffix``:


.. csv-table:: "Loss Description"
    :header: "loss_type_suffix", "loss_key", "description"

	"D", "WC", "Workers Compensation"
	"R", "PROD_LIAB","Products Liability - Occurance"
	"B", "PP_AUTO","Private Passenger Auto Liability/Medical"
	"H", "OTHR_LIAB", "Other Liability - Occurrence"
	"F", "MED_MAL", "Medical Malpractice - Claims Made"
	"C", "COM_AUTO", "Commercial Auto Liability/Medical"




A copy of the |LRDB| is distributed along with trikit, in addition to
a number of convenience functions intended to simplify database
interaction. This page details the |LRDB| convenience functions, with 
examples demonstrating typical usage scenarios. 


API
^^^^^^^^^^^^^^^^^^^^^^^^


Sample datasets are loaded using the ``load`` utility available in trikit's
top-level.


.. autofunction:: trikit.load



When ``dataset`` = "lrdb", a number of additional keyword arguments can be
passed to ``load``. For example, to retrieve the subset of workers' compensation
losses for ``grcode`` =2143:


.. code-block:: python

    >>> import trikit
    >>> wc2143 = trikit.load(dataset="lrdb", grcode=2143, loss_key="WC")
    >>> type(wc2143)
    pandas.core.frame.DataFrame

    >>> wc2143.shape
    (55, 3)


Notice that with ``grcode`` and ``loss_key`` specified as above, the
returned DataFrame contains 55 records as expected (recall that by default,
``lower_right_ind`` is set to False, otherwise the shape of the returned
DataFrame would be (100, 3).

If we repeat the same example but this time leave out the ``loss_key`` ="WC",
the arguments passed into ``load`` no longer specify a unique collection of
losses. When this occurs, one of three courses of action that can be taken,
determined by the value passed to the ``action``
parameter:

``action=None``
    Return the |LRDB| subset of data "as-is", without further aggregation
    or selection. This will result in a DataFrame in excess of 55 records
    (100 records if ``lower_right_ind=True``). This is the default.

``action="aggregate"``
    Aggregate the returned subset over "origin" and "dev". Note that setting
    ``action="aggregate"`` implicitly sets ``allcols=False``.

``action="random"``
    Select at random a single group from the remaining groups obtained
    after filtering via the original function arguments. A ``RandomState``
    object can optionally be passed to ``load``.


To demonstrate, ``load`` is called with each option and the result inspected.

First, we look at ``action=None``. Retrieve the unique "loss-key"-"grcode"
combinations from returned dataset.

.. code-block:: python

    >>> dat1 = trikit.load(dataset="lrdb", grcode=2143, action=None)
    >>> dat1[["loss_key", "grcode"]].drop_duplicates().reset_index(drop=True)
        loss_key  grcode
    0         WC    2143
    1  PROD_LIAB    2143
    2    PP_AUTO    2143
    3  OTHR_LIAB    2143
    4   COM_AUTO    2143




Next, we inspect the unique "loss-key"-"grcode" combinations returned
when ``action="random"``.

.. code-block:: python

    >>> dat2 = trikit.load(dataset="lrdb", grcode=2143, action="random")
    >>> dat2[["loss_key", "grcode"]].drop_duplicates().reset_index(drop=True)
      loss_key  grcode
    0       WC    2143


Finally, we inspect the output generated when ``action="aggregate"``. Note that
when this option is given, "loss_key" and "grcode" are not included in
the output, since ``action="aggregate"`` specifies that "value" should be
aggregated over "origin" and "dev".

.. code-block:: python

    >>> dat3 = trikit.load(dataset="lrdb", grcode=2143, action="aggregate")
    >>> dat3.shape
    (55, 3)



To return all records of the |LRDB|, call ``load`` with only ``dataset="lrdb"``.

.. code-block:: python

    lrdb = trikit.load(dataset="lrdb")
    lrdb.shape
    (42845, 5)



.. autofunction:: trikit.get_lrdb_lobs


.. autofunction:: trikit.get_lrdb_groups


.. autofunction:: trikit.get_lrdb_specs







.. |LRDB| replace:: CAS Loss Reserving Database
__ https://www.casact.org/research/index.cfm?fa=loss_reserves_data


.. rubric:: Footnotes

.. [#f1] https://www.casact.org/research/index.cfm?fa=loss_reserves_data
.. [#f2] Text of the second footnote.



