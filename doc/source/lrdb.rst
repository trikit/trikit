
.. _lrdb:

================================================================================
The CAS Loss Reserving Database
================================================================================



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
	Currently limited to ``"com_auto"``, but will be expanded in a future release.
	(see mapping below)
               
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

``ep_dir``   	
	Premiums earned at incurral year - direct and assumed     

``ep_ceded``       
	Premiums earned at incurral year - ceded       
	
``ep_net``       	
	earned_prem_dir - earned_prem_ceded      

``single``         	
	**1** indicates a single entity, **0** indicates a group insurer         

``posted_reserve_97``     
	Posted reserves in year 1997 taken from the Underwriting and Investment 
	Exhibit Part 2A, including net losses unpaid and unpaid loss adjustment
	expenses     

``train_ind``	      
	**1** indicates whether the value associated with a particular 
	``origin-dev`` combination would fall in the upper-left of a typical loss 
	triangle   


The following table provides a description of the type of losses associated 
with each unique combination of ``loss_key`` and
``loss_type_suffix``:

.. note:: At present, the only available option for losas_key is ``"com_auto"``.

.. csv-table:: "Loss Description"
    :header: "loss_type_suffix", "loss_key", "description"

	"D", "wc", "Workers Compensation"
	"R", "prod_liab","Products Liability - Occurance"
	"B", "pp_auto","Private Passenger Auto Liability/Medical"
	"H", "othr_liab", "Other Liability - Occurrence"
	"F", "med_mal", "Medical Malpractice - Claims Made"
	"C", "comauto", "Commercial Auto Liability/Medical"




A copy of the |LRDB| is distributed along with trikit, in addition to
a number of convenience functions intended to simplify database
interaction. This page details the |LRDB| convenience functions.


Loss Reserve Database API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Sample datasets are loaded using the ``load`` utility available in trikit's
top-level.


.. autofunction:: trikit.load



When ``dataset="lrdb"``, a number of additional keyword arguments can be
passed to ``load``. For example, to retrieve the subset of commercial
auto losses for ``grcode=1767``::

    In [1]: from trikit import load
    In [2]: df = trikit.load(dataset="lrdb", grcode=1767, loss_key="comauto")
    In [3]: type(df)
    pandas.core.frame.DataFrame



Notice that with ``grcode`` and ``loss_key`` specified as above, the
returned DataFrame contains 55 records as expected (recall that by default,
``train_ind`` is set to False, otherwise the shape of the returned
DataFrame would be (100, 3).



.. autofunction:: trikit.get_lrdb_lobs


.. autofunction:: trikit.get_lrdb_groups


.. autofunction:: trikit.get_lrdb_specs







.. |LRDB| replace:: CAS Loss Reserving Database
__ https://www.casact.org/research/index.cfm?fa=loss_reserves_data



.. rubric:: Footnotes

.. [#f1] https://www.casact.org/research/index.cfm?fa=loss_reserves_data




