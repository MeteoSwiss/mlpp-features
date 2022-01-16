import logging

import xarray as xr

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


def aspect_500m(data, **kwargs):
    """
    Terrain aspect at a 500m scale
    """
    return (
        data["terrain"]
        .preproc.get("ASPECT_500M_SIGRATIO1")
        .preproc.interp(kwargs["points"])
        .astype("float32")
        .load()
    )
