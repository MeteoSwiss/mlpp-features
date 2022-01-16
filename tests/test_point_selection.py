import numpy as np
import pytest

import mlpp_features.point_selection as ps


COORDS_STA = [
    (8.536, 47.48),
    (10.283, 46.793),
    (8.301, 47.036),
    (8.853, 46.707),
    (9.529, 46.577),
    (8.603, 46.32),
    (7.018, 46.546),
    (6.942, 46.811),
    (7.94, 47.005),
]


def test_point_selection_regular(raw_dataset, get_point_coords):
    """Test selection of point for irregular grids"""

    grid_res_meters = 1000
    model = raw_dataset(grid_res_meters=grid_res_meters)
    selector = ps.EuclideanNearestRegular(model, "epsg:2056")
    assert selector.grid_res == pytest.approx(grid_res_meters)

    coords_points = get_point_coords(COORDS_STA)
    index = selector.query(coords_points)
    assert index.ndim == 1
    assert index.size == coords_points.shape[0]
    model_on_sta = model.stack(point=("y", "x")).isel(point=index)
    for n, coords in enumerate(COORDS_STA):
        lon = model_on_sta.lon.isel(point=n).values
        lat = model_on_sta.lat.isel(point=n).values
        assert (float(lon), float(lat)) == pytest.approx(coords, abs=0.01)


def test_point_selection_irregular(raw_dataset, get_point_coords):
    """Test selection of point for irregular grids"""

    grid_res_meters = 1000
    model = raw_dataset(grid_res_meters=grid_res_meters)
    selector = ps.EuclideanNearestIrregular(model)
    assert selector.grid_res == pytest.approx(grid_res_meters)

    coords_points = get_point_coords(COORDS_STA)
    index = selector.query(coords_points)
    assert index.ndim == 1
    assert index.size == coords_points.shape[0]
    model_on_sta = model.stack(point=("y", "x")).isel(point=index)
    for n, coords in enumerate(COORDS_STA):
        lon = model_on_sta.lon.isel(point=n).values
        lat = model_on_sta.lat.isel(point=n).values
        assert (float(lon), float(lat)) == pytest.approx(coords, abs=0.01)
