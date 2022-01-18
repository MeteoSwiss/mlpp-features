import numpy as np
import pytest

import mlpp_features.point_selection as ps


COORDS_STA = [
    ("KLO", 8.536, 47.48),
    ("SCU", 10.283, 46.793),
    ("LUZ", 8.301, 47.036),
    ("DIS", 8.853, 46.707),
    ("PMA", 9.529, 46.577),
    ("CEV", 8.603, 46.32),
    ("MLS", 7.018, 46.546),
    ("PAY", 6.942, 46.811),
    ("NAP", 7.94, 47.005),
    ("Tromso", 18.96, 69.6),  # a point far away  ...
]

SELECTORS_METHODS = [
    ps.EuclideanNearestRegular,
    ps.EuclideanNearestIrregular,
]


@pytest.mark.parametrize("selector_method", SELECTORS_METHODS)
def test_point_selection(raw_dataset, selector_method):
    """Test selection of point for grids"""

    sta_name = [sta[0] for sta in COORDS_STA]
    sta_lon = [sta[1] for sta in COORDS_STA]
    sta_lat = [sta[2] for sta in COORDS_STA]
    grid_res_meters = 1000

    model = raw_dataset(grid_res_meters=grid_res_meters)

    selector = selector_method(model, "epsg:2056")
    index, sta_mask = selector.query((sta_lon, sta_lat))
    model_on_sta = model.stack(point=("y", "x")).isel(point=index).reset_index("point")

    assert selector.grid_res == pytest.approx(grid_res_meters)
    assert index.ndim == 1
    assert len(sta_mask) == len(sta_name)
    assert index.size == len(sta_lon) - 1  # Tromso must be excluded
    assert [p for p, m in zip(sta_name, sta_mask) if not m][0] == "Tromso"
    for n, sta in enumerate(zip(sta_lon, sta_lat, sta_mask)):
        if not sta[2]:
            continue
        lon = model_on_sta.lon.isel(point=n).values
        lat = model_on_sta.lat.isel(point=n).values
        assert (float(lon), float(lat)) == pytest.approx(sta[:2], abs=0.01)
