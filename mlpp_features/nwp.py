import logging

import xarray as xr

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


def wind_speed_ensavg(data, **kwargs):
    """
    Ensemble mean of wind speed
    """
    return (
        data["nwp"]
        .preproc.get(["eastward_wind", "northward_wind"])
        .preproc.norm()
        .mean("member")
        .preproc.interp(kwargs["points"])
        .astype("float32")
        .load()
    )
