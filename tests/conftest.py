import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer


@pytest.fixture
def raw_dataset():
    """Create dataset as if loaded from zarr files, still unprocessed."""

    def _data(grid_res_meters):

        n_members = 5
        reftimes = pd.date_range("2000-01-01T00", "2000-01-02T00", periods=3)
        leadtimes = [1, 2, 3]

        # define dummy dimensions
        n_reftimes = len(reftimes)
        n_leadtimes = len(leadtimes)

        x = np.arange(2500000, 2900000, grid_res_meters)
        y = np.arange(1000000, 1400000, grid_res_meters)
        xx, yy = np.meshgrid(x, y)
        src_proj = CRS("epsg:2056")
        dst_proj = CRS("epsg:4326")
        transformer = Transformer.from_crs(src_proj, dst_proj, always_xy=True)
        lon, lat = transformer.transform(xx, yy)

        # define dummy variables
        var_shape = (n_reftimes, n_leadtimes, n_members, y.size, x.size)
        northward_wind = np.random.randn(*var_shape)
        eastward_wind = np.random.randn(*var_shape)
        wind_speed_of_gust = np.random.randn(*var_shape)

        # Create dataset
        ds = xr.Dataset(
            {
                "eastward_wind": (
                    ["reftime", "leadtime", "member", "y", "x"],
                    eastward_wind,
                ),
                "northward_wind": (
                    ["reftime", "leadtime", "member", "y", "x"],
                    northward_wind,
                ),
                "wind_speed_of_gust": (
                    ["reftime", "leadtime", "member", "y", "x"],
                    wind_speed_of_gust,
                ),
            },
            coords={
                "lat": (["y", "x"], lat),
                "lon": (["y", "x"], lon),
                "x": x,
                "y": y,
                "reftime": reftimes,
                "leadtime": leadtimes,
                "member": np.arange(n_members),
            },
        )

        # Add validtime
        ds = ds.assign_coords(
            validtime=ds.reftime + ds.leadtime.astype("timedelta64[h]")
        )

        ds.attrs.update({"crs": "epsg:2056"})

        return ds

    return _data
