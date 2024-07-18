import pandas as pd
import pytest
import xarray as xr

import mlpp_features.selectors as sel


def test_station_selection_regular(stations_dataframe, terrain_dataset):
    """Test selection of station for regular grids"""

    grid_res_meters = 1000

    stations = stations_dataframe(outlier=True)
    terrain = terrain_dataset(grid_res_meters=grid_res_meters)

    selector = sel.EuclideanNearestRegular(terrain)
    assert selector.grid_res == pytest.approx(grid_res_meters)

    index = selector.query(stations)
    assert index.dtype == "int"
    assert isinstance(index, xr.DataArray)
    assert index.ndim == 1
    assert "station" in index.dims
    assert index.size == len(stations)

    index = index.where(index.valid, drop=True).astype(int)  # where() casts to float
    assert index.size == len(stations) - 1  # Tromso must be excluded

    terrain_on_sta = terrain.stack(point=("y", "x")).isel(point=index)


def test_station_selection_irregular(stations_dataframe, nwp_dataset):
    """Test selection of station for irregular grids"""

    grid_res_meters = 1000

    stations = stations_dataframe(outlier=True)
    model = nwp_dataset(grid_res_meters=grid_res_meters)

    selector = sel.EuclideanNearestIrregular(model)
    assert selector.grid_res == pytest.approx(grid_res_meters)

    index = selector.query(stations)
    assert index.dtype == "int"
    assert isinstance(index, xr.DataArray)
    assert index.ndim == 1
    assert "station" in index.dims
    assert index.size == len(stations)

    index = index.where(index.valid, drop=True).astype(int)  # where() casts to float
    assert index.size == len(stations) - 1  # Tromso must be excluded

    model_on_sta = model.stack(point=("y", "x")).isel(point=index)
    assert "2_9999" not in model_on_sta.station
    assert "2_9999" in stations.index
    for name, coords in stations.iterrows():
        if name == "2_9999":
            continue
        longitude = model_on_sta.longitude.sel(station=name).values
        latitude = model_on_sta.latitude.sel(station=name).values
        latlon_ref = coords[["latitude", "longitude"]]
        assert float(latitude) == pytest.approx(latlon_ref[0], abs=0.01)
        assert float(longitude) == pytest.approx(latlon_ref[1], abs=0.01)


def test_station_selection_sparse(stations_dataframe, obs_dataset):
    """Test selection of station for sparse dataset"""
    stations = stations_dataframe(outlier=True)
    obs = obs_dataset()
    selector = sel.EuclideanNearestSparse(obs)
    index = selector.query(stations, k=5)
    assert index.dtype == "int"
    assert isinstance(index, xr.DataArray)
    assert index.ndim == 2
    assert "station" in index.dims
    assert "neighbor_rank" in index.dims
    assert index.shape == (len(stations), 5)
    assert index.distance.mean() < 1e6  # TODO: ?
