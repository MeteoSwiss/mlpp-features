from inspect import getmembers, isfunction

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mlpp_features  # type: ignore


def stations_df_from_obs_ds(obs_dataset):
    stations = obs_dataset[
        ["station", "longitude", "latitude", "elevation"]
    ].to_pandas()
    return stations


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


def test_variable_euclidean_nearest_k(obs_dataset):

    k = 5
    obs_dataset = obs_dataset()[["wind_speed"]]
    stations = stations_df_from_obs_ds(obs_dataset)
    obs_nearest_k = obs_dataset.preproc.euclidean_nearest_k(stations, k)
    assert list(obs_nearest_k.dims) == ["time", "station", "neighbor_rank"]

    obs_at_klo_original = obs_dataset.wind_speed.sel(station="KLO").values
    obs_at_klo_rank_zero = obs_nearest_k.wind_speed.sel(
        station="KLO", neighbor_rank=0
    ).values
    np.testing.assert_equal(obs_at_klo_original, obs_at_klo_rank_zero)

    obs_at_rank_one_for_klo_original = obs_dataset.wind_speed.loc[
        :, obs_nearest_k.neighbor.loc["KLO", 1]
    ].values
    obs_at_rank_one_for_klo = obs_nearest_k.wind_speed.loc[:, "KLO", 1].values
    np.testing.assert_equal(obs_at_rank_one_for_klo_original, obs_at_rank_one_for_klo)


def test_variable_select_rank(obs_dataset):

    k = 5
    rank = 0

    obs_dataset = obs_dataset()[["wind_speed"]]
    stations = stations_df_from_obs_ds(obs_dataset)
    obs_nearest_k = obs_dataset.preproc.euclidean_nearest_k(stations, k)
    obs_best_rank = obs_nearest_k.preproc.select_rank(rank)
    assert list(obs_best_rank.dims) == ["time", "station"]

    # check that missing samples are subsituted by a following ranked neighbor
    nan_idx = np.argwhere(np.isnan(obs_dataset.wind_speed.values))
    non_zero_rank_idx = np.argwhere(obs_best_rank.neighbor_rank.values != 0)
    assert np.all(nan_idx == non_zero_rank_idx)

    # check that the subsitute samples correspond to the original ones
    nan_idx = list(
        zip(
            obs_dataset.time[nan_idx[:, 0]].values,
            obs_dataset.station[nan_idx[:, 1]].values,
        )
    )
    nan_subs = (
        obs_best_rank.stack(sample=["time", "station"]).sel(sample=nan_idx).wind_speed
    )
    nan_subs_idx = list(zip(nan_subs.time.values, nan_subs.neighbor.values))
    nan_sub_original = (
        obs_dataset.stack(sample=["time", "station"])
        .sel(sample=nan_subs_idx)
        .wind_speed
    )
    np.testing.assert_equal(nan_subs.values, nan_sub_original.values)
