from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer


@pytest.fixture
def stations_dataframe():
    def _data():
        stations = pd.DataFrame(
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
        return stations.set_index("name")

    return _data


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

        variables = ["wind_speed", "wind_from_direction", "wind_speed_of_gust"]
        stations = np.arange(6, dtype=int)
        names = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
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
        nan_idx = [np.random.randint(0, d, 60) for d in var_shape]
        measurements[nan_idx[0], nan_idx[1], nan_idx[2]] = np.nan
        plausibility = np.random.randn(*var_shape)

        ds = xr.Dataset(
            {
                "measurement": (["variable", "time", "station_id"], measurements),
                "plausibility": (["variable", "time", "station_id"], plausibility),
            },
            coords={
                "variable": variables,
                "time": times,
                "station_id": stations,
                "station_name": ("station_id", names),
                "station_lon": ("station_id", longitude),
                "station_lat": ("station_id", latitude),
                "station_height": ("station_id", z),
            },
        )
        return ds

    return _data


@pytest.fixture
def preproc_dataset():
    def _data():
        reftimes = pd.date_range("2000-01-01T00", "2000-01-02T00", periods=3)
        leadtimes = [timedelta(hours=t) for t in range(0, 49)]
        stations = ["OTL", "GVE", "KLO"]

        # define dummy dimensions
        n_reftimes = len(reftimes)
        n_leadtimes = len(leadtimes)
        n_stations = len(stations)

        test_ds = xr.Dataset(
            coords={
                "station": ("station", stations),
                "forecast_reference_time": ("forecast_reference_time", reftimes),
                "t": ("t", leadtimes),
            },
            data_vars={
                "foo": (
                    "station",
                    np.random.randn(n_stations),
                ),
                "bar": (
                    ("station", "forecast_reference_time", "t"),
                    np.random.randn(n_stations, n_reftimes, n_leadtimes),
                ),
            },
        )
        test_ds = test_ds.transpose("forecast_reference_time", "t", "station")
        return test_ds.astype("float32", casting="same_kind")

    return _data
