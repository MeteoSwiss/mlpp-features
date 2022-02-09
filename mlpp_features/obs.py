import logging
from typing import Dict

import xarray as xr

from mlpp_features.decorators import asarray

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@asarray
def wind_speed(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    return (
        data["obs"]
        .preproc.get("wind_speed")
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_speed_euclidean_nearest_1(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    return (
        data["obs"]
        .preproc.get("wind_speed")
        .preproc.euclidean_nearest_k(stations, k=5)
        .preproc.select_rank(rank=1)
        .preproc.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_speed_euclidean_nearest_2(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    return (
        data["obs"]
        .preproc.get("wind_speed")
        .preproc.euclidean_nearest_k(stations, k=5)
        .preproc.select_rank(rank=2)
        .preproc.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_speed_of_gust_euclidean_nearest_1(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    return (
        data["obs"]
        .preproc.get("wind_speed_of_gust")
        .preproc.euclidean_nearest_k(stations, k=5)
        .preproc.select_rank(rank=1)
        .preproc.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_speed_of_gust_euclidean_nearest_2(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    return (
        data["obs"]
        .preproc.get("wind_speed_of_gust")
        .preproc.euclidean_nearest_k(stations, k=5)
        .preproc.select_rank(rank=2)
        .preproc.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_from_direction_gust_euclidean_nearest_1(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    return (
        data["obs"]
        .preproc.get("wind_speed_of_gust")
        .preproc.euclidean_nearest_k(stations, k=5)
        .preproc.select_rank(rank=1)
        .preproc.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_from_direction_gust_euclidean_nearest_2(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    return (
        data["obs"]
        .preproc.get("wind_speed_of_gust")
        .preproc.euclidean_nearest_k(stations, k=5)
        .preproc.select_rank(rank=2)
        .preproc.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )
