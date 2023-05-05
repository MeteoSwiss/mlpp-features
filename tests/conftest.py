from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer


def _stations_dataframe(outlier=False):
    stations = pd.DataFrame(
        [
            ("BAS", 7.583, 47.541, 328, 1.0, 12.0, 0.0),
            ("LUG", 8.960, 46.004, 310, 1.0, 10.0, 27.34),
            ("GVE", 6.122, 46.248, 426, 1.0, 10.0, 0.0),
            ("GUT", 9.279, 47.602, 452, 1.0, 12.0, 0.0),
            ("KLO", 8.536, 47.48, 428, 1.0, 10.5, 0.0),
            ("SCU", 10.283, 46.793, 1306, 1.0, 10.0, 0.0),
            ("LUZ", 8.301, 47.036, 456, 1.0, 8.41, 32.51),
            ("DIS", 8.853, 46.707, 1199, 1.0, 10.0, 0.0),
            ("PMA", 9.529, 46.577, 2670, 1.0, 10.0, 0.0),
            ("CEV", 8.603, 46.32, 421, 1.0, 10.0, 6.85),
            ("MLS", 7.018, 46.546, 1976, 1.0, 10.0, 13.31),
            ("PAY", 6.942, 46.811, 491, 1.0, 10.0, 0.0),
            ("NAP", 7.94, 47.005, 1406, 1.0, 15.0, 0.0),
        ],
        columns=[
            "station",
            "longitude",
            "latitude",
            "elevation",
            "owner_id",
            "pole_height",
            "roof_height",
        ],
    )
    if outlier:
        # a station far away  ...
        outlier_station = pd.DataFrame(
            [
                {
                    "station": "Tromso",
                    "longitude": 18.96,
                    "latitude": 69.6,
                    "elevation": np.nan,
                    "owner_id": 99.0,
                    "pole_height": np.nan,
                    "roof_height": np.nan,
                }
            ]
        )
        stations = pd.concat((stations, outlier_station), ignore_index=True)
    return stations.set_index("station")


@pytest.fixture
def stations_dataframe():
    def _data(outlier=False):
        return _stations_dataframe(outlier)

    return _data


@pytest.fixture
def nwp_dataset():
    """Create dataset as if loaded from zarr files, still unprocessed."""

    def _data(grid_res_meters, var_names=None):

        n_members = 2
        reftimes = pd.date_range("2000-01-01T00", "2000-01-02T00", periods=3)
        leadtimes = np.array([0, 1, 2, 3], dtype="timedelta64[h]")

        # define dummy dimensions
        n_reftimes = len(reftimes)
        n_leadtimes = len(leadtimes)

        x = np.arange(480000, 840000, grid_res_meters)
        y = np.arange(75000, 300000, grid_res_meters)
        xx, yy = np.meshgrid(x, y)
        src_proj = CRS("epsg:21781")
        dst_proj = CRS("epsg:4326")
        transformer = Transformer.from_crs(src_proj, dst_proj, always_xy=True)
        longitude, latitude = transformer.transform(xx, yy)

        # define dummy variables
        var_shape = (n_reftimes, n_leadtimes, n_members, y.size, x.size)
        if var_names is None:
            var_names = [
                "air_temperature",
                "atmosphere_boundary_layer_thickness",
                "dew_point_temperature",
                "duration_of_sunshine",
                "eastward_wind",
                "northward_wind",
                "specific_humidity",
                "surface_air_pressure",
                "surface_downwelling_longwave_flux_in_air",
                "surface_diffuse_downwelling_shortwave_flux_in_air",
                "surface_upwelling_shortwave_flux_in_air",
                "surface_direct_downwelling_shortwave_flux_in_air",
                "wind_speed_of_gust",
            ]

        # Create dataset
        ds = xr.Dataset(
            None,
            coords={
                "latitude": (["y", "x"], latitude),
                "longitude": (["y", "x"], longitude),
                "x": x,
                "y": y,
                "forecast_reference_time": reftimes,
                "t": leadtimes,
                "realization": np.arange(n_members),
                "HSURF": (["y", "x"], np.random.randn(y.size, x.size)),
            },
        )

        # Add variables
        for var in var_names:
            ds[var] = (
                ["forecast_reference_time", "t", "realization", "y", "x"],
                np.random.randn(*var_shape).astype(np.float32),
            )

        # Add valid time coordinate
        ds = ds.assign_coords(
            time=ds.forecast_reference_time + ds.t.astype("timedelta64[h]")
        )

        ds.attrs.update({"crs": "epsg:4326"})

        return ds

    return _data


@pytest.fixture
def terrain_dataset():
    """Create dataset as if loaded from zarr files, still unprocessed."""

    def _data(grid_res_meters, var_names=None):

        x = np.arange(480000, 840000, grid_res_meters)
        y = np.arange(75000, 300000, grid_res_meters)

        # define dummy variables
        var_shape = (y.size, x.size)
        if var_names is None:
            var_names = [
                "ASPECT_500M_SIGRATIO1",
                "ASPECT_2000M_SIGRATIO1",
                "DEM",
                "SLOPE_500M_SIGRATIO1",
                "SLOPE_2000M_SIGRATIO1",
                "SN_DERIVATIVE_500M_SIGRATIO1",
                "SN_DERIVATIVE_2000M_SIGRATIO1",
                "STD_500M",
                "STD_2000M",
                "TPI_500M",
                "TPI_2000M",
                "VALLEY_NORM_2000M_SMTHFACT0.5",
                "VALLEY_NORM_20000M_SMTHFACT0.5",
                "VALLEY_DIR_2000M_SMTHFACT0.5",
                "WE_DERIVATIVE_500M_SIGRATIO1",
                "WE_DERIVATIVE_2000M_SIGRATIO1",
            ]

        # Create dataset
        ds = xr.Dataset(
            None,
            coords={
                "x": x,
                "y": y,
            },
        )

        # Add variables
        for var in var_names:
            ds[var] = (
                ["y", "x"],
                np.random.randn(*var_shape).astype(np.float32),
            )

        # add SX
        ds["SX_50M_RADIUS500"] = (
            ["y", "x", "wind_from_direction"],
            np.random.randn(y.size, x.size, 10).astype(int),
        )
        ds["wind_from_direction"] = np.arange(10)

        ds.attrs.update({"crs": "epsg:21781"})

        return ds

    return _data


@pytest.fixture
def obs_dataset():
    """Create observational dataset as if loaded from zarr files, still unprocessed."""

    def _data():

        variables = [
            "air_temperature",
            "dew_point_temperature",
            "relative_humidity",
            "surface_air_pressure",
            "water_vapor_mixing_ratio",
            "wind_speed",
            "wind_from_direction",
            "wind_speed_of_gust",
        ]

        stations = _stations_dataframe()
        times = pd.date_range("2000-01-01T00", "2000-01-02T00", freq="1H")

        n_times = len(times)
        n_stations = len(stations)

        var_shape = (n_times, n_stations)
        ds = xr.Dataset(
            None,
            coords={
                "time": times,
                "station": stations.index,
                "longitude": ("station", stations.longitude),
                "latitude": ("station", stations.latitude),
                "elevation": ("station", stations.elevation),
                "owner_id": ("station", np.random.randint(1, 5, stations.shape[0])),
                "pole_height": ("station", np.random.randint(5, 15, stations.shape[0])),
                "roof_height": ("station", np.zeros(stations.shape[0])),
            },
        )
        for var in variables:
            measurements = np.random.randn(*var_shape)
            nan_idx = [np.random.randint(0, d, 60) for d in var_shape]
            measurements[nan_idx[0], nan_idx[1]] = np.nan
            ds[var] = (("time", "station"), measurements)
        return ds

    return _data


@pytest.fixture
def preproc_dataset():
    def _data():
        reftimes = pd.date_range("2000-01-01T00", "2000-01-02T00", periods=3)
        leadtimes = [timedelta(hours=t) for t in range(0, 49)]
        stations = _stations_dataframe()

        # define dummy dimensions
        n_reftimes = len(reftimes)
        n_leadtimes = len(leadtimes)
        n_stations = len(stations)

        test_ds = xr.Dataset(
            coords={
                "station": stations.index,
                "longitude": ("station", stations.longitude),
                "latitude": ("station", stations.latitude),
                "elevation": ("station", stations.elevation),
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
