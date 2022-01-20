import logging
from typing import Dict

import xarray as xr
import numpy as np

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


def variable_euclidean_nearest_k(obs, k):
    """
    Select k nearest neighbours using euclidean distance.
    """

    distance, index = obs.preproc.selector.query(k=k)

    stations_id = obs.station_id.values
    c = [stations_id, range(index.shape[1])]
    d = ["station_id", "neighbour_rank"]

    stations_id = xr.DataArray(stations_id[index], coords=c, dims=d)
    distance = xr.DataArray(distance, coords=c, dims=d)

    return (
        obs.rename({"station_id": "neighbour_id"})
        .reset_coords(drop=True)
        .assign_coords(neighbour_distance=distance)
        .sel(neighbour_id=stations_id)
    )


def variable_select_rank(obs, rank, k):
    """
    Select the ranked observations at each timestep.
    """

    obs = obs["measurement"]

    # find index of nearest non-missing measurement at each time
    mask = ~np.isnan(obs)
    index_prefix = xr.where(
        mask.any(dim="neighbour_rank"), mask.argmax(dim="neighbour_rank"), -1
    )

    # add the requested rank to the prefix
    time_dependent_index = xr.DataArray(
        index_prefix + rank,
        coords=[obs.time, obs.station_id],
        dims=["time", "station_id"],
    )
    # make sure no index is > k
    time_dependent_index = xr.where(
        time_dependent_index <= k, time_dependent_index, k - 1
    )

    # select at each timestep
    obs = obs.isel(neighbour_rank=time_dependent_index)

    return obs.transpose("time", "station_id")


def wind_speed_euclidean_nearest_1(data: Dict[str, xr.Dataset], *args) -> xr.Dataset:
    """
    Nearest observed wind speed.
    """
    rank = 0
    k = 5

    wind_speed = data["obs"][["measurement"]].sel(variable="wind_speed")
    wind_speed_euclidean_nearest_k = variable_euclidean_nearest_k(wind_speed, k)
    wind_speed_euclidean_nearest_1 = variable_select_rank(
        wind_speed_euclidean_nearest_k, rank=rank, k=k
    )
    return wind_speed_euclidean_nearest_1.to_dataset().astype("float32")
