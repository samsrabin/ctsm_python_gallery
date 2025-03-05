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

    def setUp(self):
        self.huifrac_var = mci.DEFAULT_VAR_DICT["huifrac_var"]
        self.gddharv_var = mci.DEFAULT_VAR_DICT["gddharv_var"]

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
                self.huifrac_var: xr.DataArray(data=huifrac_in),
                self.gddharv_var: xr.DataArray(data=gddharv_in),
            }
        )
        huifrac_out = mci._get_huifrac(ds)
        self.assertTrue(np.array_equal(huifrac_out, huifrac_target))

    def test_get_isimip3_min_hui_patch0th(self):
        """
        Test that _get_isimip3_min_hui() works as expected when patch is on 0th dimension
        """
        n_patch = 4
        n_time = 2
        shape = (n_patch, n_time)
        huifrac_in = np.empty(shape)
        huifrac_in_da = xr.DataArray(data=huifrac_in, dims=["patch", "time"])
        vegstr = ["corn", "wheat", "soy", "rice"]
        vegstr_da = xr.DataArray(data=vegstr, dims=["patch"])
        ds = xr.Dataset(
            data_vars={
                self.huifrac_var: huifrac_in_da,
                "patches1d_itype_veg_str": vegstr_da,
            }
        )

        # Check that you've set things up right
        self.assertTupleEqual(huifrac_in.shape, shape)
        self.assertTrue("patch" in ds.dims)
        self.assertTrue("time" in ds.dims)

        # Expect 0.8 where corn, 0.9 elsewhere
        target = np.array([[0.8, 0.8], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]])

        result = mci._get_isimip3_min_hui(ds, self.huifrac_var)
        self.assertTrue(np.array_equal(result, target))

    def setup_minviablehui_ds_pftlast(self):
        """
        Set up Dataset and target for minimum viable HUI testing with pft in last dimension
        """
        n_pft = 4
        n_time = 2
        shape = (n_time, n_pft)
        huifrac_in = np.empty(shape)
        huifrac_in_da = xr.DataArray(data=huifrac_in, dims=["time", "pft"])
        vegstr = ["corn", "wheat", "soy", "rice"]
        vegstr_da = xr.DataArray(data=vegstr, dims=["pft"])
        ds = xr.Dataset(
            data_vars={
                self.huifrac_var: huifrac_in_da,
                "pfts1d_itype_veg_str": vegstr_da,
            }
        )

        # Check that you've set things up right
        self.assertTupleEqual(huifrac_in.shape, shape)
        self.assertTrue("pft" in ds.dims)
        self.assertTrue("time" in ds.dims)

        # Expect 0.8 where corn, 0.9 elsewhere
        target = np.array([[0.8, 0.9, 0.9, 0.9], [0.8, 0.9, 0.9, 0.9]])

        return ds, target

    def test_get_isimip3_min_hui_pftlast(self):
        """
        Test that _get_isimip3_min_hui() works as expected when pft is on last dimension
        """
        ds, target = self.setup_minviablehui_ds_pftlast()

        result = mci._get_isimip3_min_hui(ds, self.huifrac_var)
        self.assertTrue(np.array_equal(result, target))

    def test_get_min_viable_hui_number(self):
        """
        Test that _get_min_viable_hui() returns min_viable_hui if it's a number
        """
        dummy_ds = xr.Dataset()
        min_viable_hui = 0.87
        self.assertEqual(
            mci._get_min_viable_hui(dummy_ds, min_viable_hui, self.huifrac_var), min_viable_hui
        )

    def test_get_min_viable_hui_isimip3(self):
        """
        Test that _get_min_viable_hui() returns min_viable_hui if it's "isimip3"
        """
        ds, target = self.setup_minviablehui_ds_pftlast()
        self.assertTrue(
            np.array_equal(mci._get_min_viable_hui(ds, "isimip3", self.huifrac_var), target)
        )

    def test_get_min_viable_hui_ggcmi3(self):
        """
        Test that _get_min_viable_hui() returns min_viable_hui if it's "ggcmi3"
        """
        ds, target = self.setup_minviablehui_ds_pftlast()
        self.assertTrue(
            np.array_equal(mci._get_min_viable_hui(ds, "ggcmi3", self.huifrac_var), target)
        )

    def test_get_min_viable_hui_error(self):
        """
        Test that _get_min_viable_hui() errors for invalid min_viable_hui
        """
        dummy_ds = xr.Dataset()
        with self.assertRaises(NotImplementedError):
            mci._get_min_viable_hui(dummy_ds, "abc123", self.huifrac_var)
