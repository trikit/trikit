"""
Various trikit utilities. Contains convenience functions in support of
the CAS Loss Reserving Database and other sample datasets.
"""
import pandas as pd
from . import triangle



def _load(dataset, tri_type=None, dataref=None):
    """
    Load the specified sample dataset. If ``tri_type`` is not None, return sample
    dataset as specified triangle (one of ``{"cum", "incr"}``).

    Parameters
    ----------
    dataset: str
        Specifies which sample dataset to load. The complete set of sample
        datasets can be obtained by calling ``get_datasets``.

    tri_type: ``None`` or {"incr", "cum"}
        If ``None``, data subset is returned as pd.DataFrame. Otherwise,
        return subset as either incremental or cumulative triangle type.
        Default value is None.

    dataref: str
        Location of dataset reference.

    Returns
    -------
    Either pd.DataFrame, trikit.triangle.IncrTriangle or trikit.triangle.CumTriangle.
    """
    if dataset not in dataref.keys():
        raise KeyError("Specified dataset does not exist: `{}`".format(dataset))
    else:
        data_path = dataref[dataset]
        loss_data = pd.read_csv(data_path, delimiter=",")
        loss_data = loss_data[["origin", "dev", "value"]].reset_index(drop=True)

        if tri_type is not None:
            if not tri_type.startswith(("c", "i")):
                raise ValueError("tri_type must be one of {{'cum', 'incr'}}, not `{}`.".format(tri_type))
            else:
                loss_data = triangle.totri(loss_data, tri_type=tri_type)

    return(loss_data)



def _load_lrdb(tri_type=None, loss_type="incurred", lob="comauto", grcode=1767,
               grname=None, train_only=True, dataref=None):
    """
    Load the CAS Loss Reserving Database subset of losses. Additional
    keyword arguments are used to subset the CAS Loss Reserving Database to the
    records of interest.
    Within the Loss Reserving Database, "loss_key" and "grname" uniquely
    partition losses into 100 record blocks if ``lower_right_ind``=True,
    otherwise losses are partitioned into 55 record blocks. All available
    combinations of "loss_key" and "grcode" (referred to as "specs")
    can be obtained by calling ``get_lrdb_specs``.
    Note that when ``tri_type`` is "cum" or "incr", the remaining subset
    of records after applying ``lob``, ``grcode`` and ``grname`` filters will
    be aggregated into a single triangle.

    Parameters
    ----------
    tri_type: ``None`` or {"incr", "cum"}
        If ``None``, lrdb subset is returned as pd.DataFrame. Otherwise,
        return subset as either incremental or cumulative triangle type.
        Default value is None.

    lob: str
        Specifies the line of business to return. Available options are
        ``['comauto', 'ppauto', 'wkcomp', 'medmal', 'prodliab', 'othliab']``.

    grcode: str
        NAIC company code including insurer groups and single insurers.
        For a full listing, call ``get_lrdb_specs``.

    grname: str
        NAIC company name (including insurer groups and single insurers).
        The complete mapping of available grcodes can be obtained by
        calling ``get_lrdb_specs``.

    loss_type: {"paid", "incurred"}
        Specifies which loss data to load. Can be one of "paid" or
        "incurred". Defaults to "incurred". Note that bulk losses
        have already been subtracted from schedule P incurred losses.
        Applies only when ``dataset``="lrdb", otherwise parameter is
        ignored.

    train_only: bool
        If True, the upper-left portion of the triangle will be returned.
        The upper-left portion of the triangle typically consists of
        actual loss experience. If False, the completed triangle, consisting
        of 100 observations is returned. Defaults to True. Applies only when
        ``dataset`` ="lrdb", otherwise parameter is ignored.

    dataref: str
        Location of dataset reference.

    Returns
    -------
    Either pd.DataFrame, trikit.triangle.IncrTriangle or trikit.triangle.CumTriangle.
    """
    if not loss_type.startswith(("p", "i")):
        raise ValueError("loss_type should be one of {{'paid', 'incurred'}}, not `{}`.".format(loss_type))
    elif loss_type.lower().startswith("i"):
        loss_field = "incrd_loss"
    elif loss_type.lower().startswith("p"):
        loss_field = "paid_loss"

    data_path = dataref["lrdb"]
    loss_data = pd.read_csv(data_path, delimiter=",")
    loss_data = loss_data[
        ["loss_key", "grcode", "grname", "origin", "dev", loss_field, "train_ind"]
        ]

    if lob is not None:
        if lob not in loss_data["loss_key"].unique():
            raise ValueError("`{}` is not a valid lob selection.".format(lob))
        else:
            loss_data = loss_data[loss_data.loss_key == lob].reset_index(drop=True)

    if grcode is not None:
        if grcode not in loss_data["grcode"].unique():
            raise ValueError("`{}` is not a valid grcode selection.".format(grcode))
        else:
            loss_data = loss_data[loss_data.grcode == grcode].reset_index(drop=True)

    if grname is not None:
        if grname not in loss_data["grname"].unique():
            raise ValueError("`{}` is not a valid grname selection.".format(grname))
        else:
            loss_data = loss_data[loss_data.grname == grname].reset_index(drop=True)

    if train_only:
        loss_data = loss_data[loss_data.train_ind == 1].reset_index(drop=True)

    loss_data = loss_data.rename({loss_field: "value"}, axis=1)
    loss_data = loss_data[["origin", "dev", "value"]].reset_index(drop=True)

    if tri_type is not None:
        if not tri_type.startswith(("c", "i")):
            raise ValueError("tri_type should be one of {{'cum', 'incr'}}, not `{}`.".format(tri_type))
        else:
            loss_data = triangle.totri(loss_data, tri_type=tri_type)

    return(loss_data)



def _get_datasets(dataref):
    """
    Generate a list containing the names of available sample datasets.

    Parameters
    ----------
    dataref: str
        Location of dataset reference.

    Returns
    -------
    list
        Names of available sample datasets.
    """
    return(sorted([ii for ii in dataref.keys() if ii != "lrdb"]))



def _get_lrdb_lobs(lrdb_path):
    """
    Return the unique "loss_key" entries present in the CAS Loss
    Reserving Database (lrdb).

    Parameters
    ----------
    lrdb_path: str
        Location of CAS loss reserving database.

    Returns
    -------
    list
    """
    lrdb = pd.read_csv(lrdb_path, sep=",")
    lrdb = lrdb["loss_key"].unique()
    return(sorted(lrdb.tolist()))



def _get_lrdb_specs(lrdb_path):
    """
    Return a DataFrame containing the unique combinations of "loss_key",
    "grname" and "grcode" from the CAS Loss Reserving Database (lrdb).

    Parameters
    ----------
    lrdb_path: str
        Location of CAS loss reserving database.

    Returns
    -------
    pd.DataFrame
    """
    fields = ["loss_key", "grcode", "grname"]
    lrdb = pd.read_csv(lrdb_path, sep=",")
    lrdb = lrdb[fields].drop_duplicates().reset_index(drop=True)
    lrdb = lrdb.sort_values(by=["loss_key", "grcode"])
    return(lrdb)
