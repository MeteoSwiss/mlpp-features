import logging
from typing import Dict

import xarray as xr

from mlpp_features.decorators import asarray


LOGGER = logging.getLogger(__name__)


# Set global options
xr.set_options(keep_attrs=True)


@asarray
def aspect_500m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain aspect at a 500m scale
    """
    return (
        data["terrain"]
        .preproc.get("ASPECT_500M_SIGRATIO1")
        .preproc.interp(stations)
        .astype("float32")
    )


@asarray
def elevation_50m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain elevation at 50m resolution.
    """
    return data["terrain"].preproc.get("DEM").preproc.interp(stations).astype("float32")


@asarray
def tpi_500m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain TPI at a 500m scale
    """
    return (
        data["terrain"]
        .preproc.get("TPI_500M")
        .preproc.interp(stations)
        .astype("float32")
    )
