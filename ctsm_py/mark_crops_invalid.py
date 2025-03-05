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


def zero_immatures(ds, in_var="YIELD", min_viable_hui=None, mxmats=None, var_dict=DEFAULT_VAR_DICT):

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
            da_out = _mark_invalid_hui_too_low(da_out, huifrac, min_viable_hui_touse)
        da_out.attrs["min_viable_hui"] = min_viable_hui

    # Get variants with values set to 0 if season was longer than CLM PFT parameter mxmat
    if mxmat_limited:
        da_out = mark_invalid_season_too_long(ds, da_out, mxmats, var_dict["gslen_var"])

    # Save details
    if min_viable_hui or mxmat_limited:
        da_out.attrs["min_viable_hui"] = min_viable_hui
        da_out.attrs["mxmat_limited"] = mxmat_limited

    return da_out


def _mark_invalid_hui_too_low(da_in, huifrac, min_viable_hui_touse):
    tmp_da = da_in.copy()
    tmp = tmp_da.copy().values
    dont_include = (huifrac < min_viable_hui_touse) & (tmp > 0)
    tmp[np.where(dont_include)] = 0
    # if "MATURE" in out_var:
    #     tmp[np.where(~dont_include & ~np.isnan(tmp))] = 1
    #     tmp_da.attrs["units"] = "fraction"
    da_out = xr.DataArray(data=tmp, attrs=tmp_da.attrs, coords=tmp_da.coords)
    return da_out


def _get_min_viable_hui(ds, min_viable_hui, huifrac_var, huifrac):
    if min_viable_hui in ["isimip3", "ggcmi3"]:
        corn_value = 0.8
        other_value = 0.9
        min_viable_hui_touse = np.full_like(huifrac, fill_value=other_value)
        for veg_str in np.unique(ds.patches1d_itype_veg_str.values):
            if "corn" not in veg_str:
                continue
            is_thistype = np.where((ds.patches1d_itype_veg_str.values == veg_str))[0]
            patch_index = list(ds[huifrac_var].dims).index("patch")
            if patch_index == 0:
                min_viable_hui_touse[is_thistype, ...] = corn_value
            elif patch_index == ds[huifrac_var].ndim - 1:
                min_viable_hui_touse[..., is_thistype] = corn_value
            else:
                # Need patch to be either first or last dimension to allow use of ellipses
                raise RuntimeError(
                    "Temporarily rearrange min_viable_hui_touse so that patch dimension is"
                    f" first (0) or last ({ds[huifrac_var].ndim - 1}), instead of"
                    f" {patch_index}."
                )
    elif isinstance(min_viable_hui, str):
        raise RuntimeError(
            f"min_viable_hui {min_viable_hui} not recognized. Accepted strings are ggcmi3 or"
            " isimip3"
        )
    else:
        min_viable_hui_touse = min_viable_hui
    return min_viable_hui_touse


def mark_invalid_season_too_long(ds, da_in, mxmats, gslen_var):
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
        ] = 0
    da_out = xr.DataArray(data=tmp_ra, coords=da_in.coords, attrs=da_in.attrs)
    return da_out
