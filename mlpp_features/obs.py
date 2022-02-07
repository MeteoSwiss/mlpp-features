import logging
from typing import Dict

import xarray as xr
import numpy as np

from mlpp_features.decorators import asarray

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@asarray
def wind_speed(data: Dict[str, xr.Dataset], *args, **kwargs) -> xr.DataArray:
    return data["obs"][["measurement"]].sel(variable="wind_speed")


@asarray
def wind_speed_euclidean_nearest_1(data: Dict[str, xr.Dataset]) -> xr.DataArray:
    return (
        data["obs"][["measurement"]]
        .sel(variable="wind_speed")
        .preproc.euclidean_nearest_k(k=5)
        .preproc.select_rank(rank=1)
    )


@asarray
def wind_speed_euclidean_nearest_2(data: Dict[str, xr.Dataset]) -> xr.DataArray:
    return (
        data["obs"][["measurement"]]
        .sel(variable="wind_speed")
        .preproc.euclidean_nearest_k(k=5)
        .preproc.select_rank(rank=2)
    )


@asarray
def wind_speed_of_gust_euclidean_nearest_1(data: Dict[str, xr.Dataset]) -> xr.DataArray:
    return (
        data["obs"][["measurement"]]
        .sel(variable="wind_speed_of_gust")
        .preproc.euclidean_nearest_k(k=5)
        .preproc.select_rank(rank=1)
    )


@asarray
def wind_speed_of_gust_euclidean_nearest_2(data: Dict[str, xr.Dataset]) -> xr.DataArray:
    return (
        data["obs"][["measurement"]]
        .sel(variable="wind_speed_of_gust")
        .preproc.euclidean_nearest_k(k=5)
        .preproc.select_rank(rank=2)
    )


@asarray
def wind_from_direction_gust_euclidean_nearest_1(
    data: Dict[str, xr.Dataset]
) -> xr.DataArray:
    return (
        data["obs"][["measurement"]]
        .sel(variable="wind_speed_of_gust")
        .preproc.euclidean_nearest_k(k=5)
        .preproc.select_rank(rank=1)
    )


@asarray
def wind_from_direction_gust_euclidean_nearest_2(
    data: Dict[str, xr.Dataset]
) -> xr.DataArray:
    return (
        data["obs"][["measurement"]]
        .sel(variable="wind_speed_of_gust")
        .preproc.euclidean_nearest_k(k=5)
        .preproc.select_rank(rank=2)
    )
