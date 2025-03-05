"""
Module to unit-test mark_crops_invalid.py
"""

import unittest
import numpy as np
import xarray as xr

import ctsm_py.mark_crops_invalid as mci

# pylint: disable=protected-access


class TestUnitMarkCropsInvalid(unittest.TestCase):
    """
    Class to unit-test mark_crops_invalid.py
    """

    def test_pft_or_patch_patch(self):
        """
        Test that _pft_or_patch() works for patch-dimensioned Dataset
        """
        var_name = "patches1d_itype_veg_str"
        ds = xr.Dataset(data_vars={var_name: xr.DataArray})
        pftpatch_str, pftpatch_var = mci._pft_or_patch(ds)
        self.assertEqual(pftpatch_str, "patch")
        self.assertEqual(pftpatch_var, var_name)

    def test_pft_or_patch_pft(self):
        """
        Test that _pft_or_patch() works for pft-dimensioned Dataset
        """
        var_name = "pfts1d_itype_veg_str"
        ds = xr.Dataset(data_vars={var_name: xr.DataArray})
        pftpatch_str, pftpatch_var = mci._pft_or_patch(ds)
        self.assertEqual(pftpatch_str, "pft")
        self.assertEqual(pftpatch_var, var_name)

    def test_pft_or_patch_error_neither(self):
        """
        Test that _pft_or_patch() errors if neither patch nor pft is found
        """
        var_name = "abc123"
        ds = xr.Dataset(data_vars={var_name: xr.DataArray})
        with self.assertRaises(KeyError):
            mci._pft_or_patch(ds)

    def test_pft_or_patch_error_both(self):
        """
        Test that _pft_or_patch() errors if both patch and pft are found
        """
        ds = xr.Dataset(
            data_vars={
                "patches1d_itype_veg_str": xr.DataArray,
                "pfts1d_itype_veg_str": xr.DataArray,
            }
        )
        with self.assertRaises(NotImplementedError):
            mci._pft_or_patch(ds)

    def test_get_huifrac(self):
        """
        Test that _get_huifrac() replaces values as expected
        """
        huifrac_in = np.array([np.nan, 1, 0.5, 0.2])
        huifrac_target = np.array([1, 1, 0.5, 0.2])
        gddharv_in = np.array([0, 1987, 2012, 2016.4])
        ds = xr.Dataset(
            data_vars={
            mci.DEFAULT_VAR_DICT["huifrac_var"]: xr.DataArray(data=huifrac_in),
            mci.DEFAULT_VAR_DICT["gddharv_var"]: xr.DataArray(data=gddharv_in),
            }
        )
        huifrac_out = mci._get_huifrac(ds)
        self.assertTrue(np.array_equal(huifrac_out, huifrac_target))
