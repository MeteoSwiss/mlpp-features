import logging
from typing import Dict

import xarray as xr

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


def aspect_500m(data: Dict[str, xr.Dataset], *coords, **kwargs):
    """
    Terrain aspect at a 500m scale
    """
    return (
        data["terrain"]
        .preproc.get("ASPECT_500M_SIGRATIO1")
        .preproc.interp(*coords)
        .astype("float32")
    )


def elevation(data: Dict[str, xr.Dataset], *coords, **kwargs) -> xr.Dataset:
    """
    Extract height of POI from DEM
    """
    return (
        data["terrain"]
        .preproc.get("dem_50m")
        .preproc.interp(kwargs["target"], {"method": "nearest"})
        .astype("float32")
    )
