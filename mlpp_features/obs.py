import logging
from typing import Dict

import xarray as xr
import numpy as np

from mlpp_features.utils import asarray

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


def variable_euclidean_nearest_k(obs, k):
    """
    Select k nearest neighbours using euclidean distance.
    """

    obs = obs.swap_dims({"station_id": "station_name"}).rename(
        {"station_name": "point"}
    )

    distance, index = obs.preproc.selector.query(k=k)

    points = obs.point.values
    c = [points, range(index.shape[1])]
    d = ["point", "neighbour_rank"]

    stations_name = xr.DataArray(points[index], coords=c, dims=d)
    distance = xr.DataArray(distance, coords=c, dims=d)

    return (
        obs.rename({"point": "neighbour_name"})
        .reset_coords(drop=True)
        .assign_coords(neighbour_distance=distance)
        .sel(neighbour_name=stations_name)
    )


def variable_select_rank(obs, rank, k):
    """
    Select the ranked observations at each timestep.
    """

    obs = obs["measurement"].sel(neighbour_rank=slice(rank, None))

    # find index of nearest non-missing measurement at each time
    mask = ~np.isnan(obs)
    index_prefix = xr.where(
        mask.any(dim="neighbour_rank"), mask.argmax(dim="neighbour_rank"), -1
    )

    # add the requested rank to the prefix
    time_dependent_index = xr.DataArray(
        index_prefix,
        coords=[obs.time, obs.point],
        dims=["time", "point"],
    )
    # make sure no index is > k
    time_dependent_index = xr.where(
        time_dependent_index <= k, time_dependent_index, k - 1
    )

    # select at each timestep
    obs = obs.isel(neighbour_rank=time_dependent_index)

    return obs.transpose("time", "point")


def variable_euclidean_nearest(
    data: Dict[str, xr.Dataset], variable, rank: int, k=5
) -> xr.Dataset:
    """
    Nearest observed wind speed.
    """
    wind_speed = data["obs"][["measurement"]].sel(variable=variable)
    wind_speed_euclidean_nearest_k = variable_euclidean_nearest_k(wind_speed, k)
    wind_speed_euclidean_nearest_rank = variable_select_rank(
        wind_speed_euclidean_nearest_k, rank=rank, k=k
    )
    return wind_speed_euclidean_nearest_rank.astype("float32")


# wind speed
@asarray
def wind_speed_euclidean_nearest_1(data: Dict[str, xr.Dataset]) -> xr.DataArray:
    return variable_euclidean_nearest(data, "wind_speed", 1)


@asarray
def wind_speed_euclidean_nearest_2(data: Dict[str, xr.Dataset]) -> xr.DataArray:
    return variable_euclidean_nearest(data, "wind_speed", 2)


# wind speed of gust
@asarray
def wind_speed_of_gust_euclidean_nearest_1(data: Dict[str, xr.Dataset]) -> xr.DataArray:
    return variable_euclidean_nearest(data, "wind_speed_of_gust", 1)


@asarray
def wind_speed_of_gust_euclidean_nearest_2(data: Dict[str, xr.Dataset]) -> xr.DataArray:
    return variable_euclidean_nearest(data, "wind_speed_of_gust", 2)


# wind direction
@asarray
def wind_direction_euclidean_nearest_1(data: Dict[str, xr.Dataset]) -> xr.DataArray:
    return variable_euclidean_nearest(data, "wind_from_direction", 1)


@asarray
def wind_direction_euclidean_nearest_2(data: Dict[str, xr.Dataset]) -> xr.DataArray:
    return variable_euclidean_nearest(data, "wind_from_direction", 2)
