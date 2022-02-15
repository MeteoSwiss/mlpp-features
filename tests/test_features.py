from inspect import getmembers, isfunction

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mlpp_features  # type: ignore
from mlpp_features.decorators import KEEP_STA_COORDS


class TestFeatures:

    pipelines = [obj[1] for obj in getmembers(mlpp_features) if isfunction(obj[1])][1:]

    @pytest.fixture(autouse=True)
    def _make_datasets(
        self, nwp_dataset, obs_dataset, terrain_dataset, stations_dataframe
    ):
        self._nwp = nwp_dataset(1e4)
        self._obs = obs_dataset()
        self._terrain = terrain_dataset(1e4)
        self._stations = stations_dataframe()

    @pytest.mark.parametrize("pipeline,", pipelines)
    def test_raise_keyerror(self, pipeline):
        """Test that all features raise a KeyError when called with empy inputs"""
        empty_data = {
            "nwp": xr.Dataset(),
            "terrain": xr.Dataset(),
            "obs": xr.Dataset(),
        }
        with pytest.raises(KeyError):
            pipeline(empty_data, None, None, None)

    @pytest.mark.parametrize("pipeline,", pipelines)
    def test_features(self, pipeline):
        """"""
        data = {
            "nwp": self._nwp,
            "terrain": self._terrain,
            "obs": self._obs,
        }
        stations = self._stations
        reftimes = pd.date_range("2000-01-01T00", "2000-01-02T00", periods=4)
        leadtimes = list(range(3))
        da = pipeline(data, stations, reftimes, leadtimes)
        assert isinstance(da, xr.DataArray)
        assert "variable" not in da.dims
        if "t" in da.dims:
            assert da.t.dtype == int
        assert all([coord in KEEP_STA_COORDS + list(da.dims) for coord in da.coords])
        assert all([dim in da.coords for dim in da.dims])
