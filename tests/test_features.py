from inspect import getmembers, isfunction
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mlpp_features
from mlpp_features import decorators  # type: ignore
from mlpp_features.decorators import KEEP_STA_COORDS


class TestFeatures:

    pipelines = [obj[1] for obj in getmembers(mlpp_features) if isfunction(obj[1])][1:]
    pipelines = [pipe for pipe in pipelines if not pipe.__name__.startswith("_")]
    pipelines = [pipe for pipe in pipelines if not hasattr(decorators, pipe.__name__)]

    @pytest.fixture(autouse=True)
    def _make_datasets(
        self, nwp_dataset, obs_dataset, terrain_dataset, clim_dataset, stations_dataframe
    ):
        self._nwp = nwp_dataset(1e4)
        self._obs = obs_dataset()
        self._terrain = terrain_dataset(1e4)
        self._climatology = clim_dataset()
        self._stations = stations_dataframe()

    @pytest.mark.parametrize("pipeline,", pipelines)
    def test_raise_keyerror(self, pipeline):
        """Test that all features raise a KeyError when called with empty inputs"""
        empty_data = {
            "nwp": xr.Dataset(),
            "terrain": xr.Dataset(),
            "obs": xr.Dataset(),
            "climatology": xr.Dataset(),
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
            "climatology": self._climatology,
        }
        stations = self._stations
        reftimes = pd.date_range("2000-01-01T00", "2000-01-01T18", periods=4)
        leadtimes = np.arange(3).astype("timedelta64[h]")
        da = pipeline(data, stations, reftimes, leadtimes)
        assert isinstance(da, xr.DataArray)
        if np.issubdtype(da.dtype, np.number):
            assert da.dtype == "float32"
        if "forecast_reference_time" in da.dims:
            assert all([r in da.forecast_reference_time for r in reftimes])
            assert da.forecast_reference_time.dtype == np.dtype("datetime64[ns]")
        if "lead_time" in da.dims:
            assert all([lead_time in da.lead_time for lead_time in leadtimes])
            assert da.lead_time.dtype == np.dtype("timedelta64[ns]")
        assert "variable" not in da.dims
        assert all([coord in KEEP_STA_COORDS + list(da.dims) for coord in da.coords])
        assert all([dim in da.coords for dim in da.dims])
