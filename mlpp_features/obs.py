import logging
from typing import Dict

import xarray as xr
import numpy as np 

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


def wind_speed_euclidean_nearest_1(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.Dataset:
    """
    Nearest observed wind speed.
    """
    
    rank = 1

    distance, index = data["obs"].preproc.selector.query(coords[1:])
    index = xr.DataArray(index, coords=[coords[0], range(index.shape[1])], dims=['station','neighbour_rank'])
    distance = xr.DataArray(distance, coords=[coords[0], range(distance.shape[1])], dims=['station','neighbour_rank'])
    name = xr.DataArray(np.array(coords[0])[index], coords=[coords[0], range(index.shape[1])], dims=["station", "neighbour_rank"])

    neighbour_info = {
        "neighbour": name,
        # "neighbour_distance": distance
    }

    return (
        data["obs"].measurement
        .loc[["wind_speed"]]
        .to_dataset("variable")
        .rename({"station":"neighbour"}) # this is a trick to make multidimensional indexing work
        .sel(neighbour=name)
        .reset_coords(drop=True)
        .assign_coords(neighbour_info)
        .isel(neighbour_rank=rank)
        .astype("float32")
    )