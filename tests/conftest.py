from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer

from mlpp_features.terrain import ALPINE_SOUTHERN_CREST_WGS84


def _stations_dataframe(outlier=False):
    # fmt: off
    stations = pd.DataFrame(
        [
            ("BAS", "1_75", 1, 75, "BAS", "Basel / Binningen", 1, "MeteoSchweiz", 7.583, 47.541, 316.14, 12.0, 0.0),
            ("LUG", "1_47", 1, 47, "LUG", "Lugano", 1, "MeteoSchweiz", 8.960, 46.004, 272.56, 10.0, 27.34),
            ("GVE", "1_58", 1, 58, "GVE", "Gen\u00e8ve / Cointrin", 1, "MeteoSchweiz", 6.122, 46.248, 415.53, 10.0, 0.0),
            ("GUT", "1_79", 1, 79, "GUT", "G\u00fcttingen", 1, "MeteoSchweiz", 9.279, 47.602, 439.78, 12.0, 0.0),
            ("KLO", "1_59", 1, 59, "KLO", "Z\u00fcrich / Kloten", 1, "MeteoSchweiz", 8.536, 47.48, 435.92, 10.5, 0.0),
            ("SCU", "1_30", 1, 30, "SCU", "Scuol", 1, "MeteoSchweiz", 10.283, 46.793, 1304.42, 10.0, 0.0),
            ("LUZ", "1_68", 1, 68, "LUZ", "Luzern", 1, "MeteoSchweiz", 8.301, 47.036, 454.0, 8.41, 32.51),
            ("DIS", "1_54", 1, 54, "DIS", "Disentis", 1, "MeteoSchweiz", 8.853, 46.707, 1198.03, 10.0, 0.0),
            ("PMA", "1_862", 1, 862, "PMA", "Piz Martegnas", 1, "MeteoSchweiz", 9.529, 46.577, 2668.34, 10.0, 0.0),
            ("CEV", "1_843", 1, 843, "CEV", "Cevio", 1, "MeteoSchweiz", 8.603, 46.32, 420.0, 10.0, 6.85),
            ("MLS", "1_38", 1, 38, "MLS", "Le Mol\u00e9son", 1, "MeteoSchweiz", 7.018, 46.546, 1977.0, 10.0, 13.31),
            ("PAY", "1_32", 1, 32, "PAY", "Payerne", 1, "MeteoSchweiz", 6.942, 46.811, 489.17, 10.0, 0.0),
            ("NAP", "1_48", 1, 48, "NAP", "Napf", 1, "MeteoSchweiz", 7.94, 47.005, 1404.03, 15.0, 0.0),

        ],
        columns=[
            "station",
            "name",
            "type_id",
            "point_id",
            "nat_abbr",
            "fullname",
            "owner_id",
            "owner_name",
            "longitude",
            "latitude",
            "height_masl",
            "pole_height",
            "roof_height",
        ],
    )
    # fmt: on
    if outlier:
        # a station far away  ...
        outlier_station = pd.DataFrame(
            [
                {
                    "station": "Tromso",
                    "longitude": 18.96,
                    "latitude": 69.6,
                    "height_masl": np.nan,
                    "nat_abbr": np.nan,
                    "fullname": "Tromso",
                    "name": "2_9999",
                    "type_id": 2,
                    "point_id": 9999,
                    "owner_id": 9999,
                    "owner_name": np.nan,
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
                "cloud_area_fraction",
                "cloud_area_fraction_in_high_troposphere",
                "cloud_area_fraction_in_medium_troposphere",
                "cloud_area_fraction_in_low_troposphere",
                "dew_point_temperature",
                "duration_of_sunshine",
                "eastward_wind",
                "northward_wind",
                "mass_fraction_of_cloud_liquid_water_in_air",
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
                "forecast_reference_time": reftimes.astype("datetime64[ns]"),
                "lead_time": leadtimes.astype("timedelta64[ns]"),
                "realization": np.arange(n_members),
                "surface_altitude": (["y", "x"], np.random.randn(y.size, x.size)),
            },
        )

        # Add variables
        for var in var_names:
            ds[var] = (
                ["forecast_reference_time", "lead_time", "realization", "y", "x"],
                np.random.randn(*var_shape).astype(np.float32),
            )

        # Add valid time coordinate
        ds = ds.assign_coords(time=ds.forecast_reference_time + ds.lead_time)

        ds.attrs.update({"crs": "epsg:4326", "source_id": "dummy_model"})

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
                "SN_DERIVATIVE_100000M_SIGRATIO1",
                "STD_500M",
                "STD_2000M",
                "TPI_500M",
                "TPI_2000M",
                "TPI_60000M_SMTHFACT1",
                "TPI_100000M_SMTHFACT1",
                "VALLEY_NORM_1000M_SMTHFACT0.5",
                "VALLEY_NORM_2000M_SMTHFACT0.5",
                "VALLEY_NORM_10000M_SMTHFACT0.5",
                "VALLEY_NORM_20000M_SMTHFACT0.5",
                "VALLEY_DIR_1000M_SMTHFACT0.5",
                "VALLEY_DIR_2000M_SMTHFACT0.5",
                "VALLEY_DIR_10000M_SMTHFACT0.5",
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

        ds.attrs.update({"grid_mapping": str({"epsg_code": "21781"})})

        return ds

    return _data


@pytest.fixture
def obs_dataset():
    """Create observational dataset as if loaded from zarr files, still unprocessed."""

    def _data():

        variables = [
            "air_temperature",
            "cloud_area_fraction",
            "dew_point_temperature",
            "relative_humidity",
            "surface_air_pressure",
            "water_vapor_mixing_ratio",
            "wind_speed",
            "wind_from_direction",
            "wind_speed_of_gust",
        ]

        stations = _stations_dataframe()
        times = pd.date_range("2000-01-01T00", "2000-01-02T00", freq="1h")

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
                "height_masl": ("station", stations.height_masl),
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
        leadtimes = [timedelta(hours=lead_time) for lead_time in range(0, 49)]
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
                "height_masl": ("station", stations.height_masl),
                "forecast_reference_time": ("forecast_reference_time", reftimes),
                "lead_time": ("lead_time", leadtimes),
            },
            data_vars={
                "foo": (
                    "station",
                    np.random.randn(n_stations),
                ),
                "bar": (
                    ("station", "forecast_reference_time", "lead_time"),
                    np.random.randn(n_stations, n_reftimes, n_leadtimes),
                ),
            },
        )
        test_ds = test_ds.transpose("forecast_reference_time", "lead_time", "station")
        return test_ds.astype("float32", casting="same_kind")

    return _data


@pytest.fixture
def preproc_dataset_ens():
    def _data():
        reftimes = pd.date_range("2000-01-01T00", "2000-01-02T00", periods=3)
        leadtimes = [timedelta(hours=lead_time) for lead_time in range(0, 49)]
        stations = _stations_dataframe()
        realizations = list(range(3))

        # define dummy dimensions
        n_reftimes = len(reftimes)
        n_leadtimes = len(leadtimes)
        n_stations = len(stations)
        n_realizations = len(realizations)

        test_ds = xr.Dataset(
            coords={
                "station": stations.index,
                "longitude": ("station", stations.longitude),
                "latitude": ("station", stations.latitude),
                "height_masl": ("station", stations.height_masl),
                "forecast_reference_time": ("forecast_reference_time", reftimes),
                "lead_time": ("lead_time", leadtimes),
                "realization": ("realization", realizations),
            },
            data_vars={
                "bar": (
                    ("station", "forecast_reference_time", "lead_time", "realization"),
                    np.random.randn(
                        n_stations, n_reftimes, n_leadtimes, n_realizations
                    ),
                ),
            },
        )
        test_ds = test_ds.transpose(
            "forecast_reference_time", "lead_time", "station", "realization"
        )
        return test_ds.astype("float32", casting="same_kind")

    return _data


@pytest.fixture
def alpine_ridge():
    return ALPINE_SOUTHERN_CREST_WGS84


@pytest.fixture
def horizontal_ridge():
    ridge = [(46, 6.0), (46, 11.0)]
    return ridge


@pytest.fixture
def diagonal_ridge():
    ridge = [(45.5, 6.0), (47.0, 11.0)]
    return ridge
