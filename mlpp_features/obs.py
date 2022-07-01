import logging
from typing import Dict

import xarray as xr

from mlpp_features.decorators import asarray, reuse

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@reuse
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


@reuse
@asarray
def wind_speed_of_gust(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
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
