import logging
from typing import Dict

import xarray as xr

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


def wind_speed_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.Dataset:
    """
    Ensemble mean of wind speed
    """
    return (
        data["nwp"]
        .preproc.get(["eastward_wind", "northward_wind"])
        .preproc.norm()
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )
