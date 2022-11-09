from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mlpp_features  # type: ignore


def test_align_time(preproc_dataset):
    time_shift = 1
    n_reftimes = 4

    ds = preproc_dataset()

    t0 = pd.Timestamp(ds.forecast_reference_time.values[0])
    reftimes = [t0 + (1 + n) * timedelta(hours=time_shift) for n in range(n_reftimes)]
    reftimes = pd.DatetimeIndex(reftimes)
    leadtimes = np.array([0, 1], dtype="timedelta64[h]")
    ds_aligned = ds.preproc.align_time(reftimes, leadtimes)
    assert isinstance(ds_aligned, xr.Dataset)
    assert ds_aligned.sizes["forecast_reference_time"] == len(reftimes)
    assert ds_aligned.sizes["t"] == len(leadtimes)
    assert "source_leadtime" not in ds_aligned
    assert ds_aligned.forecast_reference_time.dtype == np.dtype("datetime64[ns]")
    assert ds_aligned.t.dtype == np.dtype("timedelta64[ns]")
    ds_aligned.sel(forecast_reference_time=reftimes)
    ds_aligned.sel(t=leadtimes)

    for var, da in ds.items():
        array_aligned = ds_aligned.isel(t=0, forecast_reference_time=0)[var].values
        array_original = da.isel(
            t=time_shift, forecast_reference_time=0, missing_dims="ignore"
        ).values
        np.testing.assert_array_equal(array_aligned, array_original)


def test_align_time_dims(preproc_dataset):
    time_shift = 12
    n_reftimes = 3

    ds = preproc_dataset()

    t0 = pd.Timestamp(ds.forecast_reference_time.values[0])
    reftimes = [t0 + n * timedelta(hours=time_shift) for n in range(n_reftimes)]
    reftimes = pd.DatetimeIndex(reftimes)
    leadtimes = [0, 1]
    ds_aligned = ds.preproc.align_time(
        reftimes, leadtimes, return_source_leadtimes=True
    )
    assert "forecast_reference_time" in ds_aligned.source_leadtime.dims


def test_unstack_time(obs_dataset):

    ds = obs_dataset()
    t0 = pd.Timestamp(ds.time.values[0])
    reftimes = pd.date_range(t0, t0 + timedelta(hours=12), freq="3H")
    leadtimes = np.array([0, 1, 2], dtype="timedelta64[h]")
    ds_unstacked = ds.preproc.unstack_time(reftimes, leadtimes)
    assert isinstance(ds_unstacked, xr.Dataset)
    assert ds_unstacked.sizes["forecast_reference_time"] == len(reftimes)
    assert ds_unstacked.sizes["t"] == len(leadtimes)
    assert ds_unstacked.forecast_reference_time.dtype == np.dtype("datetime64[ns]")
    assert ds_unstacked.t.dtype == np.dtype("timedelta64[ns]")
    ds_unstacked.sel(forecast_reference_time=reftimes)
    ds_unstacked.sel(t=leadtimes)

    for var, da in ds.items():
        array_unstacked = ds_unstacked.isel(forecast_reference_time=0)[var].values
        array_original = da.isel(time=slice(None, 3)).values
        np.testing.assert_array_equal(array_unstacked, array_original)


def test_persist_observations(obs_dataset):

    ds = obs_dataset()
    t0 = pd.Timestamp(ds.time.values[0])
    reftimes = pd.date_range(t0, t0 + timedelta(hours=3), freq="1H")
    leadtimes = np.array([0, 1, 2], dtype="timedelta64[h]")
    ds_persisted = ds.preproc.persist_observations(reftimes, leadtimes)
    assert isinstance(ds_persisted, xr.Dataset)
    assert ds_persisted.sizes["forecast_reference_time"] == len(reftimes)
    assert ds_persisted.sizes["t"] == len(leadtimes)
    assert ds_persisted.forecast_reference_time.dtype == np.dtype("datetime64[ns]")
    assert ds_persisted.t.dtype == np.dtype("timedelta64[ns]")
    ds_persisted.sel(forecast_reference_time=reftimes)
    ds_persisted.sel(t=leadtimes)

    for var, da in ds.items():
        array_persisted = ds_persisted[var].values
        axis_t = list(ds_persisted[var].dims).index("t")
        np.testing.assert_array_equal(
            array_persisted.min(axis=axis_t), array_persisted.max(axis=axis_t)
        )
        array_original = da.isel(time=slice(None, len(reftimes))).values
        np.testing.assert_array_equal(array_persisted.min(axis=axis_t), array_original)


def test_daystat(preproc_dataset):
    reduction_dims = ["forecast_reference_time", "t"]
    ds = preproc_dataset()
    time = ds.forecast_reference_time + ds.t
    ds = ds.assign_coords(time=time)

    day_one = (ds.time <= ds.time[0, 0] + np.timedelta64(1, "D")) & (
        ds.time > ds.time[0, 0]
    )

    daymax = ds.preproc.daystat(xr.Dataset.max)
    daymax_day_one = (
        daymax.where(day_one, drop=True)
        .isel(forecast_reference_time=0, t=1, drop=True)
        .bar
    )

    day_one_max = (
        ds.where(day_one, drop=True)
        .isel(forecast_reference_time=0, drop=True)
        .max("t")
        .bar
    )

    xr.testing.assert_equal(day_one_max, daymax_day_one)


def test_interp(stations_dataframe, nwp_dataset):

    stations = stations_dataframe()
    ds = nwp_dataset(grid_res_meters=1000)
    ds_interp = ds.preproc.interp(stations)

    assert isinstance(ds_interp, xr.Dataset)
    assert (ds_interp.station.values == stations.index).all()
    for coord, coords in stations.items():
        assert (ds_interp[coord].values == coords).all()


def test_euclidean_nearest_k(stations_dataframe, obs_dataset):
    k = 5
    stations = stations_dataframe()
    obs = obs_dataset()
    obs_nearest_k = obs.preproc.euclidean_nearest_k(stations, k)
    assert isinstance(obs_nearest_k, xr.Dataset)
    assert obs_nearest_k.dims["neighbor_rank"] == k
    assert isinstance(obs_nearest_k, xr.Dataset)
    assert (obs_nearest_k.station.values == stations.index).all()
    for coord, coords in stations.iteritems():
        assert (obs_nearest_k[coord].values == coords).all()
    assert list(obs_nearest_k.dims) == ["time", "station", "neighbor_rank"]

    obs_at_klo_original = obs.wind_speed.sel(station="KLO").values
    obs_at_klo_rank_zero = obs_nearest_k.wind_speed.sel(
        station="KLO", neighbor_rank=0
    ).values
    np.testing.assert_equal(obs_at_klo_original, obs_at_klo_rank_zero)

    obs_at_rank_one_for_klo_original = obs.wind_speed.loc[
        :, obs_nearest_k.neighbor.loc["KLO", 1]
    ].values
    obs_at_rank_one_for_klo = obs_nearest_k.wind_speed.loc[:, "KLO", 1].values
    np.testing.assert_equal(obs_at_rank_one_for_klo_original, obs_at_rank_one_for_klo)


def test_select_rank(stations_dataframe, obs_dataset):
    rank = 0
    k = 5
    stations = stations_dataframe()
    obs = obs_dataset()
    obs_nearest_k = obs.preproc.euclidean_nearest_k(stations, k)
    with pytest.raises(ValueError):
        obs_nearest_k.preproc.select_rank(k)
    with pytest.raises(ValueError):
        obs_nearest_k.preproc.select_rank(rank)
    obs_nearest = obs_nearest_k[["wind_speed"]].preproc.select_rank(rank)
    assert isinstance(obs_nearest, xr.Dataset)
    assert (obs_nearest.station.values == stations.index).all()
    for coord, coords in stations.iteritems():
        assert (obs_nearest[coord].values == coords).all()
        assert f"neighbor_{rank}_{coord}" in obs_nearest.coords
    assert list(obs_nearest.dims) == ["time", "station"]

    # check that missing samples are subsituted by a following ranked neighbor
    nan_idx = np.argwhere(np.isnan(obs.wind_speed.values))
    non_zero_rank_idx = np.argwhere(obs_nearest[f"neighbor_{rank}_rank"].values != rank)
    assert np.all(nan_idx == non_zero_rank_idx)

    # check that the subsitute samples correspond to the original ones
    nan_idx = list(
        zip(
            obs.time[nan_idx[:, 0]].values,
            obs.station[nan_idx[:, 1]].values,
        )
    )
    nan_subs = (
        obs_nearest.stack(sample=["time", "station"]).sel(sample=nan_idx).wind_speed
    )
    nan_subs_idx = list(zip(nan_subs.time.values, nan_subs.neighbor.values))
    nan_sub_original = (
        obs.stack(sample=["time", "station"]).sel(sample=nan_subs_idx).wind_speed
    )
    np.testing.assert_equal(nan_subs.values, nan_sub_original.values)
