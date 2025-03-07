"""
Functions to generate useful crop variables not saved by CTSM
"""

import numpy as np

from ctsm_py.crop_defaults import DEFAULT_VAR_DICT


def _handle_huifrac_where_gddharv_notpos(da_huifrac, da_gddharv):
    # Error if any GDDHARV value is negative
    if np.any(da_gddharv < 0):
        raise NotImplementedError("How should negative GDDHARV affect HUIFRAC?")

    huifrac = da_huifrac.values

    # If harvest threshold HUI is 0, mark huifrac as 1
    huifrac[np.where(da_gddharv.values == 0)] = 1
    return huifrac


def get_huifrac(ds, var_dict=DEFAULT_VAR_DICT):
    """
    Given a dataset, calculate HUIFRAC as hui_var/gddharv_var
    """
    hui_var = var_dict["hui_var"]
    gddharv_var = var_dict["gddharv_var"]

    da_hui = ds[hui_var]
    da_gddharv = ds[gddharv_var]
    da_huifrac = da_hui / da_gddharv

    # Handle HUIFRAC where GDDHARV (denominator) is zero or negative
    huifrac = _handle_huifrac_where_gddharv_notpos(da_huifrac, da_gddharv)
    da_huifrac.data = huifrac

    da_huifrac.attrs["units"] = "Fraction of required"
    return da_huifrac
