import logging
from typing import Dict

import numpy as np
import xarray as xr

from mlpp_features.decorators import out_format


LOGGER = logging.getLogger(__name__)


# Set global options
xr.set_options(keep_attrs=True)


@out_format()
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


@out_format()
def aspect_2000m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain aspect at a 2000m scale
    """
    return (
        data["terrain"]
        .preproc.get("ASPECT_2000M_SIGRATIO1")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
def cos_valley_index_2000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate cosine of valley index 2km resolution
    """
    norm_valley = (
        data["terrain"]
        .preproc.get("VALLEY_NORM_2000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_2000M_SMTHFACT0.5": "cos_valley"})
    )
    dir_valley = np.cos(
        2 * np.pi / 180 * data["terrain"].preproc.get("VALLEY_DIR_2000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_2000M_SMTHFACT0.5": "cos_valley"})
    cos_valley = norm_valley * dir_valley
    cos_valley.attrs.update(data["terrain"].attrs)
    return cos_valley.preproc.interp(stations).astype("float32")


@out_format()
def elevation_50m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain elevation at 50m resolution.
    """
    return data["terrain"].preproc.get("DEM").preproc.interp(stations).astype("float32")


@out_format()
def slope_500m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Extract slope at 500m resolution
    """
    return (
        data["terrain"]
        .preproc.get("SLOPE_500M_SIGRATIO1")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
def slope_2000m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Extract slope at 2000m resolution
    """
    return (
        data["terrain"]
        .preproc.get("SLOPE_2000M_SIGRATIO1")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
def sn_derivative_500m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Extract S-N derivative at 500m resolution
    """
    return (
        data["terrain"]
        .preproc.get("SN_DERIVATIVE_500M_SIGRATIO1")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
def sn_derivative_2000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Extract S-N derivative at 2000m resolution
    """
    return (
        data["terrain"]
        .preproc.get("SN_DERIVATIVE_2000M_SIGRATIO1")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
def std_500m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain STD at a 500m scale
    """
    return (
        data["terrain"]
        .preproc.get("STD_500M")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
def std_2000m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain STD at a 2000m scale
    """
    return (
        data["terrain"]
        .preproc.get("STD_2000M")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
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


@out_format()
def tpi_2000m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain TPI at a 2000m scale
    """
    return (
        data["terrain"]
        .preproc.get("TPI_2000M")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
def valley_norm_2000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Extract valley norm 2km resolution
    """
    return (
        data["terrain"]
        .preproc.get("VALLEY_NORM_2000M_SMTHFACT0.5")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
def valley_norm_20000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Extract valley norm 20km resolution
    """
    return (
        data["terrain"]
        .preproc.get("VALLEY_NORM_20000M_SMTHFACT0.5")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
def we_derivative_500m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Extract W-E derivative at 500m resolution
    """
    return (
        data["terrain"]
        .preproc.get("WE_DERIVATIVE_500M_SIGRATIO1")
        .preproc.interp(stations)
        .astype("float32")
    )


@out_format()
def we_derivative_2000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Extract W-E derivative at 2000m resolution
    """
    return (
        data["terrain"]
        .preproc.get("WE_DERIVATIVE_2000M_SIGRATIO1")
        .preproc.interp(stations)
        .astype("float32")
    )
