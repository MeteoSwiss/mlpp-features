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
        longitude, latitude = transformer.transform(xx, yy)

        # define dummy variables
        var_shape = (n_reftimes, n_leadtimes, n_members, y.size, x.size)
        northward_wind = np.random.randn(*var_shape)
        eastward_wind = np.random.randn(*var_shape)
        wind_speed_of_gust = np.random.randn(*var_shape)

        # Create dataset
        ds = xr.Dataset(
            {
                "eastward_wind": (
                    ["forecast_reference_time", "t", "realization", "y", "x"],
                    eastward_wind,
                ),
                "northward_wind": (
                    ["forecast_reference_time", "t", "realization", "y", "x"],
                    northward_wind,
                ),
                "wind_speed_of_gust": (
                    ["forecast_reference_time", "t", "realization", "y", "x"],
                    wind_speed_of_gust,
                ),
            },
            coords={
                "latitude": (["y", "x"], latitude),
                "longitude": (["y", "x"], longitude),
                "x": x,
                "y": y,
                "forecast_reference_time": reftimes,
                "t": leadtimes,
                "realization": np.arange(n_members),
            },
        )

        # Add validtime
        ds = ds.assign_coords(
            validtime=ds.forecast_reference_time + ds.t.astype("timedelta64[h]")
        )

        ds.attrs.update({"crs": "epsg:2056"})

        return ds

    return _data


@pytest.fixture
def raw_obs_dataset():
    """Create observational dataset as if loaded from zarr files, still unprocessed."""

    def _data():

        variables = ["wind_speed", "wind_from_direction","wind_speed_of_gust"]
        stations = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        times = pd.date_range("2000-01-01T00", "2000-01-02T00", periods=10)

        n_variables = len(variables)
        n_times = len(times)
        n_stations = len(stations)

        x = np.random.uniform(2500000, 2800000, n_stations)
        y = np.random.uniform(1080000, 1300000, n_stations)
        z = np.random.uniform(100, 1000, n_stations)

        src_proj = CRS("epsg:2056")
        dst_proj = CRS("epsg:4326")
        transformer = Transformer.from_crs(src_proj, dst_proj, always_xy=True)
        longitude, latitude = transformer.transform(x, y)

        # define dummy variables
        var_shape = (n_variables, n_times, n_stations)
        measurements = np.random.randn(*var_shape)
        plausibility = np.random.randn(*var_shape)

        ds = xr.Dataset(
            {
                "measurement": (
                    ["variable", "time", "station"],
                    measurements
                ),
                "plausibility": (
                    ["variable", "time", "station"],
                    plausibility
                )
            },
            coords = {
                "variable": variables,
                "time": times,
                "station": stations,
                "station_lon": ("station", longitude),
                "station_lat": ("station", latitude),
                "station_height": ("station", z)
            }
        )

        return ds
    return _data