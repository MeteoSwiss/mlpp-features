import pandas as pd
import pytest
import xarray as xr

import mlpp_features.selectors as sel


STATIONS = pd.DataFrame(
    [
        ("KLO", 8.536, 47.48),
        ("SCU", 10.283, 46.793),
        ("LUZ", 8.301, 47.036),
        ("DIS", 8.853, 46.707),
        ("PMA", 9.529, 46.577),
        ("CEV", 8.603, 46.32),
        ("MLS", 7.018, 46.546),
        ("PAY", 6.942, 46.811),
        ("NAP", 7.94, 47.005),
        ("Tromso", 18.96, 69.6),  # a station far away  ...
    ],
    columns=["name", "longitude", "latitude"],
)
STATIONS = STATIONS.set_index("name")

GRID_SELECTORS_METHODS = [
    sel.EuclideanNearestRegular,
    sel.EuclideanNearestIrregular,
]

SPARSE_SELECTORS_METHODS = [
    sel.EuclideanNearestSparse,
]


@pytest.mark.parametrize("selector_method", GRID_SELECTORS_METHODS)
def test_station_selection(raw_dataset, selector_method):
    """Test selection of station for grids"""

    grid_res_meters = 1000

    model = raw_dataset(grid_res_meters=grid_res_meters)

    selector = selector_method(model, "epsg:2056")
    assert selector.grid_res == pytest.approx(grid_res_meters)

    index = selector.query(STATIONS)
    assert index.dtype == "int"
    assert isinstance(index, xr.DataArray)
    assert index.ndim == 1
    assert "station" in index.dims
    assert index.size == len(STATIONS)

    index = index.where(index.valid, drop=True).astype(int)  # where() casts to float
    assert index.size == len(STATIONS) - 1  # Tromso must be excluded

    model_on_sta = model.stack(point=("y", "x")).isel(point=index)
    assert "Tromso" not in model_on_sta.station
    for name, coords in STATIONS.iterrows():
        if name == "Tromso":
            continue
        longitude = model_on_sta.longitude.sel(station=name).values
        latitude = model_on_sta.latitude.sel(station=name).values
        assert (float(longitude), float(latitude)) == pytest.approx(coords, abs=0.01)


@pytest.mark.parametrize("selector_method", SPARSE_SELECTORS_METHODS)
def test_obs_station_selection(raw_obs_dataset, selector_method):
    """Test selection of station for sparse dataset"""
    obs = raw_obs_dataset()
    selector = selector_method(obs)
    index = selector.query(STATIONS, k=5)
    assert index.dtype == "int"
    assert isinstance(index, xr.DataArray)
    assert index.ndim == 2
    assert "station" in index.dims
    assert "neighbor_rank" in index.dims
    assert index.shape == (len(STATIONS), 5)
    assert index.distance.mean() < 1e6  # TODO: ?
