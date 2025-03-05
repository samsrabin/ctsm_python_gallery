"""
Functions to mark crop seasons as invalid
"""

from types import MappingProxyType
import numpy as np
import xarray as xr

# MappingProxyType makes this dict immutable
DEFAULT_VAR_DICT = MappingProxyType(
    {
        "huifrac_var": "HUIFRAC",
        "gddharv_var": "GDDHARV",
        "gslen_var": "GSLEN",
    }
)


def _get_min_viable_hui(ds, min_viable_hui, huifrac_var, huifrac):
    """
    Get minimum viable HUI values.

    Parameters:
    ds (xarray.Dataset): Input dataset.
    min_viable_hui (float or str): Minimum viable HUI value or a string identifier.
    huifrac_var (str): Variable name for HUI fraction.
    huifrac (numpy.ndarray): HUI fraction values.

    Returns:
    numpy.ndarray: Minimum viable HUI values to use.
    """
    if min_viable_hui in ["isimip3", "ggcmi3"]:
        min_viable_hui_touse = _get_isimip3_min_hui(ds, huifrac_var, huifrac)
    elif isinstance(min_viable_hui, str):
        raise RuntimeError(
            f"min_viable_hui {min_viable_hui} not recognized. Accepted strings are ggcmi3 or"
            " isimip3"
        )
    else:
        min_viable_hui_touse = min_viable_hui
    return min_viable_hui_touse


def _get_isimip3_min_hui(ds, huifrac_var, huifrac):
    corn_value = 0.8   # Lower than other crops to account for silage maize harvest
    other_value = 0.9

    if "patches1d_itype_veg_str" in ds:
        pftpatch_str = "patch"
        pftpatch_var = "patches1d_itype_veg_str"
    elif "pfts1d_itype_veg_str" in ds:
        pftpatch_str = "pft"
        pftpatch_var = "pfts1d_itype_veg_str"
    else:
        raise KeyError("Neither patches1d_itype_veg_str nor pfts1d_itype_veg_str found in ds")

    min_viable_hui_touse = np.full_like(huifrac, fill_value=other_value)
    for veg_str in np.unique(ds[pftpatch_var].values):
        if "corn" not in veg_str:
            # Skip, because the min_viable_hui_touse array is already set to other_value there
            continue
        is_thistype = np.where((ds[pftpatch_var].values == veg_str))[0]
        pftpatch_index = list(ds[huifrac_var].dims).index(pftpatch_str)
        if pftpatch_index == 0:
            min_viable_hui_touse[is_thistype, ...] = corn_value
        elif pftpatch_index == ds[huifrac_var].ndim - 1:
            min_viable_hui_touse[..., is_thistype] = corn_value
        else:
            # Need patch to be either first or last dimension to allow use of ellipses
            raise NotImplementedError(
                f"Temporarily rearrange min_viable_hui_touse so that {pftpatch_str} dimension is"
                f" first (0) or last ({ds[huifrac_var].ndim - 1}), instead of"
                f" {pftpatch_index}."
            )
    return min_viable_hui_touse


def mark_invalid_hui_too_low(da_in, huifrac, min_viable_hui_touse, invalid_value=0):
    """
    Mark yields as invalid where HUI is too low.

    Parameters:
    da_in (xarray.DataArray): Input DataArray.
    huifrac (numpy.ndarray): HUI fraction values.
    min_viable_hui_touse (numpy.ndarray): Minimum viable HUI values to use.

    Returns:
    xarray.DataArray: DataArray with invalid yields set to zero.
    """
    tmp_da = da_in.copy()
    tmp = tmp_da.copy().values
    dont_include = (huifrac < min_viable_hui_touse) & (tmp > 0)
    tmp[np.where(dont_include)] = invalid_value
    # if "MATURE" in out_var:
    #     tmp[np.where(~dont_include & ~np.isnan(tmp))] = 1
    #     tmp_da.attrs["units"] = "fraction"
    da_out = xr.DataArray(data=tmp, attrs=tmp_da.attrs, coords=tmp_da.coords)
    return da_out


def mark_invalid_season_too_long(ds, da_in, mxmats, gslen_var, invalid_value=0):
    """
    Mark invalid yields where season length is too long.

    Parameters:
    ds (xarray.Dataset): Input dataset.
    da_in (xarray.DataArray): Input DataArray.
    mxmats (dict): Dictionary of maximum allowed season length. Format: {"crop": value}.
    gslen_var (str): Variable name for growing season length.

    Returns:
    xarray.DataArray: DataArray with invalid yields set to zero.
    """
    tmp_ra = da_in.copy().values
    for veg_str in np.unique(ds.patches1d_itype_veg_str.values):
        mxmat_veg_str = veg_str.replace("soybean", "temperate_soybean").replace(
            "tropical_temperate", "tropical"
        )
        mxmat = mxmats[mxmat_veg_str]
        tmp_ra[
            np.where(
                (ds.patches1d_itype_veg_str.values == veg_str) & (ds[gslen_var].values > mxmat)
            )
        ] = invalid_value
    da_out = xr.DataArray(data=tmp_ra, coords=da_in.coords, attrs=da_in.attrs)
    return da_out


def mark_crops_invalid(
    ds, in_var="YIELD", min_viable_hui=None, mxmats=None, var_dict=DEFAULT_VAR_DICT, invalid_value=0
):  # pylint: disable=too-many-positional-arguments
    """
    Mark a variable as invalid where minimum viable HUI wasn't reached or season was longer than
    maximum allowed length.

    Parameters:
    ds (xarray.Dataset): Input dataset.
    in_var (str): Variable name for yield. Default is "YIELD".
    min_viable_hui (float or str): Minimum viable HUI value or a string identifier.
    mxmats (dict): Dictionary of maximum allowed season length. Format: {"crop": value}.
    var_dict (dict): Dictionary of variable names.

    Returns:
    xarray.DataArray: DataArray with invalid yields set to zero.
    """
    mxmat_limited = bool(mxmats)

    da_out = ds[in_var].copy()

    # Set yield to zero where minimum viable HUI wasn't reached
    if min_viable_hui is not None:
        huifrac = ds[var_dict["huifrac_var"]].copy().values
        huifrac[np.where(ds[var_dict["gddharv_var"]].values == 0)] = 1
        min_viable_hui_touse = _get_min_viable_hui(
            ds, min_viable_hui, var_dict["huifrac_var"], huifrac
        )
        if np.any(huifrac < min_viable_hui_touse):
            da_out = mark_invalid_hui_too_low(
                da_out, huifrac, min_viable_hui_touse, invalid_value=invalid_value
            )
        da_out.attrs["min_viable_hui"] = min_viable_hui

    # Get variants with values set to 0 if season was longer than CLM PFT parameter mxmat
    if mxmat_limited:
        da_out = mark_invalid_season_too_long(
            ds, da_out, mxmats, var_dict["gslen_var"], invalid_value=invalid_value
        )

    # Save details
    if min_viable_hui or mxmat_limited:
        da_out.attrs["min_viable_hui"] = min_viable_hui
        da_out.attrs["mxmat_limited"] = mxmat_limited

    return da_out
