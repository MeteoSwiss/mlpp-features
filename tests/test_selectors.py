import pandas as pd
import pytest
import xarray as xr

import mlpp_features.selectors as sel


GRID_SELECTORS_METHODS = [
    sel.EuclideanNearestRegular,
    sel.EuclideanNearestIrregular,
]

SPARSE_SELECTORS_METHODS = [
    sel.EuclideanNearestSparse,
]


@pytest.mark.parametrize("selector_method", GRID_SELECTORS_METHODS)
def test_station_selection(stations_dataframe, raw_dataset, selector_method):
    """Test selection of station for grids"""

    grid_res_meters = 1000

    stations = stations_dataframe()
    model = raw_dataset(grid_res_meters=grid_res_meters)

    selector = selector_method(model, "epsg:2056")
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
    assert "Tromso" not in model_on_sta.station
    for name, coords in stations.iterrows():
        if name == "Tromso":
            continue
        longitude = model_on_sta.longitude.sel(station=name).values
        latitude = model_on_sta.latitude.sel(station=name).values
        latlon_ref = coords[["latitude", "longitude"]]
        assert (float(latitude), float(longitude)) == pytest.approx(
            latlon_ref, abs=0.01
        )


@pytest.mark.parametrize("selector_method", SPARSE_SELECTORS_METHODS)
def test_obs_station_selection(stations_dataframe, raw_obs_dataset, selector_method):
    """Test selection of station for sparse dataset"""
    stations = stations_dataframe()
    obs = raw_obs_dataset()
    selector = selector_method(obs)
    index = selector.query(stations, k=5)
    assert index.dtype == "int"
    assert isinstance(index, xr.DataArray)
    assert index.ndim == 2
    assert "station" in index.dims
    assert "neighbor_rank" in index.dims
    assert index.shape == (len(stations), 5)
    assert index.distance.mean() < 1e6  # TODO: ?
