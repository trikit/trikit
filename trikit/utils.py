"""
Various trikit utilities. Contains convenience functions in support of
the CAS Loss Reserving Database and other sample datasets.
"""
import sys
import os.path
import numpy as np
import pandas as pd
from numpy.random import RandomState





def _load(dataref):
    """
    trikit's sample dataset loading utility.
    """
    def func(dataset, loss_type="paid", lob=None, grcode=None,
             grname=None, upper_left_ind=True, lower_right_ind=False,
             allcols=False, action=None, random_state=None):
        """
        Load the specified dataset, returning a DataFrame of incremental
        losses. If ``dataset`` ="lrdb", additional keyword arguments are used
        to subset the CAS Loss Reserving Database to the records of interest.
        Within the Loss Reserving Database, "loss_key" and "grname" uniquely
        partition losses into 100 record blocks if ``lower_right_ind`` =True,
        otherwise losses are partitioned into 55 record blocks. All available
        combinations of "loss_key" and "grcode" (referred to as "specs")
        can be obtained by calling ``get_lrdb_specs``.
        If ``dataset`` is something other than "lrdb", then only the name of
        the target dataset as a string is required.

        Parameters
        ----------
        dataset: str
            Specifies which sample dataset to load. The complete set of sample
            datasets can be obtained by calling ``get_datasets``.

        lob: str
            One of "WC", "COM_AUTO", "MED_MAL", "OTHR_LIAB", "PP_AUTO" or
            "PROD_LIAB". When ``dataset`` ="lrdb", specifies which losses to
            target. The complete mapping of available lobs can be obtained by
            calling ``get_lrdb_lobs``. Applies only when ``dataset`` ="lrdb",
            otherwise parameter is ignored.

        grcode: str
            NAIC company code including insurer groups and single insurers.
            The complete mapping of available grcodes can be obtained by
            calling ``get_lrdb_groups``. Applies only when
            ``dataset`` ="lrdb", otherwise parameter is ignored.

        grname: str
            NAIC company name (including insurer groups and single insurers).
            The complete mapping of available grcodes can be obtained by
            calling ``get_lrdb_groups``. Applies only when
            ``dataset`` ="lrdb", otherwise parameter is ignored.

        loss_type: str
            Specifies which loss data to load. Can be one of "paid" or
            "incurred". Defaults to "paid". Applies only when
            ``dataset`` ="lrdb", otherwise parameter is ignored.

        upper_left_ind: bool
            If True, the upper-left portion of the triangle will be returned.
            The upper-left portion of the triangle typically consists of
            actual loss experience. Defaults to True. Applies only when
            ``dataset`` ="lrdb", otherwise parameter is ignored.

        lower_right_ind: bool
            If True, the lower-right portion of the triangle will be returned.
            The CAS Loss Reserve Database includes 10 development lags for
            each loss year, the intention being that the data comprising the
            lower-right portion of the triangle can be used for model
            validation purposes. Defaults to False. Applies only when
            ``dataset`` ="lrdb", otherwise parameter is ignored.

        allcols: bool
            If True, the returned DataFrame contains all columns that comprise
            the CAS Loss Reserving Database. Defaults to False. When
            ``allcols`` =False, only the columns required to convert the
            dataset to an ``IncrTriangle`` or ``CumTriangle`` instance, namely
            "origin", "dev" and "value", with the field specified by ``loss``
            renamed to "value" will be returned. Note that if
            ``action`` ="aggregate", ``allcols`` is bound to False,
            regardless of how it was originally specified in the function
            call.

        action: str
            Action to perform if the specified subsetting parameters do not
            reduce to a single "loss_key"-"grcode". If more than one
            combination of "loss_key" and "grcode" remains after applying all
            subsetting specifications, the remaining records can either be
            (1) aggregated (``action`` ="aggregate"), (2) a single
            "loss_key"-"grcode" combination can be selected at random and the
            associated records returned (``action`` ="random") or (3) the
            remaining records can be returned as-is without additional
            processing (``action`` =None). Defaults to None. Note that
            ``action``="aggregate" implicitly sets ``allcols`` to False,
            regardless of how the parameter may have been set initially.
            Applies only when ``dataset`` ="lrdb", otherwise parameter is
            ignored.

        random_state: RandomState/int
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random. Applies only when
            ``dataset ="lrdb"``, otherwise parameter is ignored.

        Returns
        -------
        pd.DataFrame
            Sample dataset or subset of the CAS Loss Reserving Database.
        """
        try:
            dataset  = dataset.lower()
            datapath = dataref[dataset]
            dat_init = pd.read_csv(datapath, delimiter=",")

        except KeyError:
            print("Specified dataset does not exist: `{}`".format(dataset))

        # Additional filtering/subsetting if dataset="lrdb".
        if dataset=="lrdb":
            if   loss_type.lower().startswith("i"): loss_field = "incrd_loss"
            elif loss_type.lower().startswith("p"): loss_field = "paid_loss"
            elif loss_type.lower().startswith("b") : loss_field = "bulk_loss"

            basecols = ["loss_key", "grcode", "origin", "dev", loss_field]
            lower_right_spec = 1 if lower_right_ind==True else 0
            upper_left_spec  = 1 if upper_left_ind==True  else 0
            loss_key_spec, loss_type_spec = lob, loss_type
            grname_spec, grcode_spec = grname, grcode

            if all([lower_right_spec, upper_left_spec]):
                dat_init = dat_init[
                    (dat_init["lower_right_ind"]==lower_right_spec) |
                    (dat_init["upper_left_ind"]==upper_left_spec)
                    ]
            elif upper_left_spec==1 and lower_right_spec==0:
                dat_init = dat_init[
                    dat_init["upper_left_ind"]==upper_left_spec
                    ]
            elif upper_left_spec==0 and lower_right_spec==1:
                dat_init = dat_init[
                    dat_init["lower_right_ind"]==lower_right_spec
                    ]
            else:
                raise ValueError(
                    "At least one of upper_left_ind or lower_right_ind must be 1."
                    )

            if loss_key_spec is not None:
                dat_init = dat_init[dat_init["loss_key"]==loss_key_spec]

            if grname_spec is not None:
                dat_init = dat_init[dat_init["grname"]==grname_spec]

            if grcode_spec is not None:
                dat_init = dat_init[dat_init["grcode"]==grcode_spec]

            # Check whether provided filter specs filter down to a single
            # loss_key-grname/grcode combination. If more than one spec
            # remains, look to `action` parameter.
            fields = ["loss_key", "grcode"]
            remaining_specs = dat_init[fields].drop_duplicates().reset_index(drop=True)

            if remaining_specs.shape[0] > 1:

                if action and action.lower().startswith("agg"):
                    # Aggregate remaining records over `origin` and `dev`.
                    # Requires dropping `loss_key` and `grcode`.
                    dat = dat_init[["origin", "dev", loss_field]].groupby(["origin", "dev"],
                        as_index=False).sum().reset_index(drop=True)

                elif action and action.lower().startswith("rand"):
                    # Check random_state and initialize random number generator.
                    if random_state is not None:
                        if isinstance(random_state, int):
                            prng = RandomState(random_state)
                        elif isinstance(random_state, RandomState):
                            prng = random_state
                    else:
                        prng = RandomState()

                    # Randomly select record from remaining_specs.
                    keepspec = remaining_specs.loc[prng.choice(remaining_specs.index)]
                    keep_loss_key, keep_grcode = keepspec.loss_key, keepspec.grcode

                    # Filter dat_init using keep_*-prefixed fields.
                    dat = dat_init[
                        (dat_init.loss_key==keep_loss_key) & (dat_init.grcode==keep_grcode)
                        ]

                    dat = dat[basecols]

                else: # Return dataset "as-is", containing > 1 specs.
                    dat = dat_init[basecols]
            else:
                dat = dat_init[basecols]

        else: # Specified dataset is not "lrdb".
            loss_field = "value"
            basecols = ["origin", "dev", "value"]
            dat      = dat_init

        if not allcols:
            dat = dat.rename({loss_field:"value"}, axis=1)
        return(dat.reset_index(drop=True))
    return(func)





# Loss Reserving Database utility functions ==================================]
def _get_datasets(dataref:dict):
    def func():
        """
        Generate a list containing the names of available sample datasets.

        Returns
        -------
        list
            Names of available sample datasets.
        """
        return(list(dataref.keys()))
    return(func)



def _get_lrdb_lobs(lrdb_path:str):
    """
    Return the unique "loss_key" entries present in the CAS Loss
    Reserving Database (lrdb).
    """
    lrdb = pd.read_csv(lrdb_path, sep=",")
    lrdb = lrdb["loss_key"].unique()
    def func():
        """
        Return unique "loss_key" entries from the CAS Loss Reserving
        Database.

        Returns
        -------
        list
            List containing unique "loss_key" entries.
        """
        return(lrdb.tolist())
    return(func)



def _get_lrdb_groups(lrdb_path:str):
    """
    Return "grcode"-"grname" mapping present in the CAS Loss
    Reserving Database (lrdb).
    """
    fields = ["grcode", "grname"]
    lrdb   = pd.read_csv(lrdb_path, sep=",")
    lrdb   = lrdb[fields].drop_duplicates().reset_index(drop=True)
    def func(returnas=pd.DataFrame):
        """
        Return lrdb groups mapping in accordance with object class
        specified in ``returnas`` argument.

        Parameters
        ----------
        returnas: type
            The object class used to represent the lrdb "grcode"-"grname"
            mapping. Valid types include pd.DataFrame, list or dict.
            Any other object passed to ``returnas`` raises TypeError. Default
            type is pd.DataFrame.

        Returns
        -------
        list, dict, pd.DataFrame
            Unique combinations of "grcode"-"grname" as present in the CAS
            Loss Reserving Database.
        """
        if returnas==pd.DataFrame:
            groups = lrdb
        elif returnas==list:
            groups_init = {j:i for i, j in set(zip(lrdb.grname, lrdb.grcode))}
            groups = list(groups_init.items())
        elif returnas==dict:
            groups = {j:i for i, j in set(zip(lrdb.grname, lrdb.grcode))}
        else:
            raise TypeError("Invalid returnas object class: `{}`".format(returnas))
        return(groups)
    return(func)



def _get_lrdb_specs(lrdb_path:str):
    """
    Return a DataFrame containing the unique combinations of "loss_key",
    "grname" and "grcode" from the CAS Loss Reserving Database (lrdb).
    """
    fields = ["loss_key", "grcode", "grname"]
    lrdb = pd.read_csv(lrdb_path, sep=",")
    lrdb = lrdb[fields].drop_duplicates().reset_index(drop=True)
    lrdb = lrdb.sort_values(by=["loss_key","grcode"])
    def func():
        """
        Return a DataFrame containing the unique combinations of "loss_key",
        "grname" and "grcode" from the CAS Loss Reserving Database (lrdb).

        Returns
        -------
        pd.DataFrame
            Unique combinations of "loss_key", "grname" and "grcode"
            from the CAS Loss Reserving Database.
        """
        return(lrdb)
    return(func)
