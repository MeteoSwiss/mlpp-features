import logging
from typing import Dict

import numpy as np
import xarray as xr

from mlpp_features.decorators import asarray, reuse

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@asarray
def air_temperature(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed air temperature in °C
    """
    return (
        data["obs"]
        .preproc.get("air_temperature")
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def dew_point_depression(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed dew point depression (T - T_d)
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    t_d = dew_point_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return (t - t_d).astype("float32")


@asarray
def dew_point_temperature(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed dew point temperature in °C
    """
    return (
        data["obs"]
        .preproc.get("dew_point_temperature")
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def surface_air_pressure(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed surface pressure in hPa
    """
    return (
        data["obs"]
        .preproc.get("surface_air_pressure")
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def relative_humidity(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed relative humidity in %
    """
    return (
        data["obs"]
        .preproc.get("relative_humidity")
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def water_vapor_mixing_ratio(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed water vapor mixing ratio in g/kg
    """
    return (
        data["obs"]
        .preproc.get("water_vapor_mixing_ratio")
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def cos_wind_from_direction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Cosine of observed wind directions
    """
    return (
        data["obs"]
        .preproc.get("wind_from_direction")
        .pipe(lambda x: x * 2 * np.pi / 360)  # to radians
        .pipe(np.cos)
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def sin_wind_from_direction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Sine of observed wind directions
    """
    return (
        data["obs"]
        .preproc.get("wind_from_direction")
        .pipe(lambda x: x * 2 * np.pi / 360)  # to radians
        .pipe(np.sin)
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def wind_speed(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed wind speed in m/s
    """
    return (
        data["obs"]
        .preproc.get("wind_speed")
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def wind_speed_of_gust(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed wind gust in m/s
    """
    return (
        data["obs"]
        .preproc.get("wind_speed_of_gust")
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def nearest_wind_speed(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed wind speed at the nearest (euclidean distance) station
    """
    return (
        data["obs"]
        .preproc.get("wind_speed")
        .preproc.euclidean_nearest_k(stations, k=5)
        .preproc.select_rank(rank=1)
        .preproc.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def distance_to_nearest_wind_speed(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Distance (euclidean, in meters) from the nearest wind speed measurement.
    """
    return (
        data["obs"]
        .preproc.get("wind_speed")
        .preproc.euclidean_nearest_k(stations, k=5)
        .preproc.select_rank(rank=1)
        .drop_vars("wind_speed")
        .reset_coords("neighbor_1_distance")
        .preproc.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def nearest_wind_speed_of_gust(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed wind gust at the nearest (euclidean distance) station
    """
    return (
        data["obs"]
        .preproc.get("wind_speed_of_gust")
        .preproc.euclidean_nearest_k(stations, k=5)
        .preproc.select_rank(rank=1)
        .preproc.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def distance_to_nearest_wind_speed_of_gust(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Distance (euclidean, in meters) from the nearest wind gust measurement.
    """
    return (
        data["obs"]
        .preproc.get("wind_speed_of_gust")
        .preproc.euclidean_nearest_k(stations, k=5)
        .preproc.select_rank(rank=1)
        .drop_vars("wind_speed_of_gust")
        .reset_coords("neighbor_1_distance")
        .preproc.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )
