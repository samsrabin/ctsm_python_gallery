"""
Module to unit-test mark_crops_invalid.py
"""

import unittest
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
