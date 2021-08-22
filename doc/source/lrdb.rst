
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


**loss_key**
	Currently limited to ``"com_auto"``, but will be expanded in a future release.
	(see mapping below)
               
**grcode**
	NAIC company code (including insurer groups and single insurers)               

**grname**
	NAIC company name (including insurer groups and single insurers)           
  
**origin**
	Accident year          

**dev**
	Development year                 

**incrd_loss**
	Incurred losses and allocated expenses reported at year end       

**paid_loss**
	paid losses and allocated expenses at year end       

**ep_dir**
	Premiums earned at incurral year - direct and assumed     

**ep_ceded**
	Premiums earned at incurral year - ceded       
	
**ep_net**
	earned_prem_dir - earned_prem_ceded      

**single**
	**1** indicates a single entity, **0** indicates a group insurer         

**posted_reserve_97**
	Posted reserves in year 1997 taken from the Underwriting and Investment 
	Exhibit Part 2A, including net losses unpaid and unpaid loss adjustment
	expenses     

**train_ind**
	**1** indicates whether the value associated with a particular 
	``origin-dev`` combination would fall in the upper-left of a typical loss 
	triangle   


The following table provides a description of the type of losses associated 
with each unique combination of ``loss_key`` and
``loss_type_suffix``:



.. csv-table:: Loss Description
    :header: "loss_type_suffix", "loss_key", "description"

	"D", "wc", "Workers Compensation"
	"R", "prod_liab","Products Liability - Occurance"
	"B", "pp_auto","Private Passenger Auto Liability/Medical"
	"H", "othr_liab", "Other Liability - Occurrence"
	"F", "med_mal", "Medical Malpractice - Claims Made"
	"C", "comauto", "Commercial Auto Liability/Medical"




A copy of the |LRDB| is distributed along with trikit, in addition to
a number of convenience functions intended to simplify database
interaction. In what follows, |LRDB| convenience functions are detailed.


Loss Reserve Database API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: load_lrdb(tri_type=None, loss_type="incurred", lob="comauto", grcode=1767,
                        grname=None, train_only=True)

    Load the CAS Loss Reserving Database subset of losses. Additional
	keyword arguments are used to subset the CAS Loss Reserving Database to the
	records of interest.
	Within the Loss Reserving Database, "lob" and "grcode" uniquely
	partition losses into 100 record blocks if ``lower_right_ind=True``,
	otherwise losses are partitioned into 55 record blocks. All available
	combinations of "lob" and "grcode" (referred to as "specs")
	can be obtained by calling ``get_lrdb_specs``.
	Note that when ``tri_type`` is "cum" or "incr", the remaining subset
	of records after applying ``lob``, ``grcode`` and ``grname`` filters will
	be aggregated into a single triangle of the specified type.

	:param tri_type: If ``None``, lrdb subset is returned as pd.DataFrame. Otherwise,
    return subset as either incremental or cumulative triangle type. Default value is None.
    :type tri_type: {None, "incr", "cum"}

	:param lob: Specifies the line of business to return. Available options are
    ``['comauto', 'ppauto', 'wkcomp', 'medmal', 'prodliab', 'othliab']``. Default
    value is "comauto".
    :type lob: str

	:param grcode: NAIC company code including insurer groups and single insurers.
    For a full listing, call ``get_lrdb_specs``. Default value is ``1767``.
    :type grcode: int

	:param grname: NAIC company name (including insurer groups and single insurers).
    The complete mapping of available grcodes can be obtained by calling ``get_lrdb_specs``.
    Default value is None.
    :type grname: str

	:param loss_type: Specifies which loss data to load. Can be one of "paid" or
    "incurred". Defaults to "incurred". Note that bulk losses have already been subtracted
    from schedule P incurred losses. Default value is "incurred".
    :type loss_type: {"paid", "incurred"}

	:param train_only: If True, the upper-left portion of the triangle will be returned.
    The upper-left portion of the triangle typically consists of actual loss experience. If
    False, the squared triangle, consisting of 100 observations is returned. Default value
    is True.
    :type train_only: bool

    :return: Either pd.DataFrame, trikit.triangle.IncrTriangle or trikit.triangle.CumTriangle.


|LRDB} datasets are loaded via ``load_lrdb``. A number of additional keyword arguments can be
passed to ``load_lrdb``. For example, to retrieve the subset of commercial auto losses for
``grcode=1767``::

    In [1]: from trikit import load_lrdb
    In [2]: df = load(grcode=1767, lob="comauto")
    In [3]: type(df)
    pandas.core.frame.DataFrame



Notice that with ``grcode`` and ``lob`` specified as above, the
returned DataFrame contains 55 records as expected (recall that by default,
``train_ind`` is set to False, otherwise the shape of the returned
DataFrame would be (100, 3).


|LRDB| lines of business can be listed by calling ``get_lrdb_lobs``::

    In [1]: from trikit import get_lrdb_lobs
    In [2]: get_lrdb_lobs()
    Out[2]: ['comauto', 'ppauto', 'wkcomp', 'medmal', 'prodliab', 'othliab']



Unique combinations of "loss_key", "grname" and "grcode" can be listed by calling ``get_lrdb_specs``::

        In [1]: from trikit import get_lrdb_specs
        In [2]: get_lrdb_specs()
        Out[2]:
            loss_key  grcode                               grname
        0    comauto     266              Public Underwriters Grp
        1    comauto     337                   California Cas Grp
        2    comauto     353                       Celina Mut Grp
        3    comauto     388                   Federal Ins Co Grp
        4    comauto     460                      Buckeye Ins Grp
        ..       ...     ...                                  ...
        431   wkcomp   41580                    Red Shield Ins Co
        432   wkcomp   42439                Toa-Re Ins Co Of Amer
        433   wkcomp   43915                       Rainier Ins Co
        434   wkcomp   44091  Dowa Fire & Marine Ins Co Ltd Us Br
        435   wkcomp   44300                   Tower Ins Co Of NY





.. |LRDB| replace:: CAS Loss Reserving Database
__ https://www.casact.org/research/index.cfm?fa=loss_reserves_data



.. rubric:: Footnotes

.. [#f1] https://www.casact.org/research/index.cfm?fa=loss_reserves_data




