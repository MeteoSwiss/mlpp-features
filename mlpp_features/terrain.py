import logging
from typing import Dict

import numpy as np
import xarray as xr

from mlpp_features.decorators import out_format
from mlpp_features import experimental as exp


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
def cos_valley_index_1000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate cosine of valley index 1km resolution
    """
    norm_valley = (
        data["terrain"]
        .preproc.get("VALLEY_NORM_1000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_1000M_SMTHFACT0.5": "cos_valley"})
    )
    dir_valley = np.cos(
        2 * np.pi / 180 * data["terrain"].preproc.get("VALLEY_DIR_1000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_1000M_SMTHFACT0.5": "cos_valley"})
    cos_valley = norm_valley * dir_valley
    cos_valley.attrs.update(data["terrain"].attrs)
    return cos_valley.preproc.interp(stations).astype("float32")


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
def cos_valley_index_10000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate cosine of valley index 10km resolution
    """
    norm_valley = (
        data["terrain"]
        .preproc.get("VALLEY_NORM_10000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_10000M_SMTHFACT0.5": "cos_valley"})
    )
    dir_valley = np.cos(
        2 * np.pi / 180 * data["terrain"].preproc.get("VALLEY_DIR_10000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_10000M_SMTHFACT0.5": "cos_valley"})
    cos_valley = norm_valley * dir_valley
    cos_valley.attrs.update(data["terrain"].attrs)
    return cos_valley.preproc.interp(stations).astype("float32")


@out_format()
def distance_to_alpine_ridge(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Compute horizontal distance to the main Alpine ridge

    **Experimental feature, use with caution!**
    """
    if all([len(ds) == 0 for ds in data.values()]):
        raise KeyError()
    alpine_crest_wgs84 = [
        [45.67975, 6.88306],
        [45.75149, 6.80643],
        [45.88912, 7.07724],
        [45.86909, 7.17029],
        [46.25074, 8.03064],
        [46.47280, 8.38946],
        [46.55972, 8.55968],
        [46.56318, 8.80080],
        [46.61256, 8.96059],
        [46.49712, 9.17104],
        [46.50524, 9.33031],
        [46.39905, 9.69325],
        [46.40885, 10.01963],
        [46.63982, 10.29218],
        [46.83630, 10.50783],
        [46.90567, 11.09742],
    ]
    points = [(lat, lon) for lat, lon in zip(stations.latitude, stations.longitude)]
    points_proj = exp.reproject_points(points, "epsg:2056")
    line_proj = exp.reproject_points(alpine_crest_wgs84, "epsg:2056")
    distances = exp.distances_points_to_line(points_proj, line_proj)
    return xr.Dataset(
        coords={
            "station": stations.index,
            "longitude": ("station", stations.longitude),
            "latitude": ("station", stations.latitude),
            "elevation": ("station", stations.elevation),
        },
        data_vars={
            "distance_to_alpine_ridge": (
                "station",
                distances,
            ),
        },
    ).astype("float32")


@out_format()
def elevation_50m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain elevation at 50m resolution.
    """
    return data["terrain"].preproc.get("DEM").preproc.interp(stations).astype("float32")


@out_format()
def sin_valley_index_1000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate sine of valley index 1km resolution
    """
    norm_valley = (
        data["terrain"]
        .preproc.get("VALLEY_NORM_1000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_1000M_SMTHFACT0.5": "sin_valley"})
    )
    dir_valley = np.sin(
        2 * np.pi / 180 * data["terrain"].preproc.get("VALLEY_DIR_1000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_1000M_SMTHFACT0.5": "sin_valley"})
    sin_valley = norm_valley * dir_valley
    sin_valley.attrs.update(data["terrain"].attrs)
    return sin_valley.preproc.interp(stations).astype("float32")


@out_format()
def sin_valley_index_2000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate sine of valley index 2km resolution
    """
    norm_valley = (
        data["terrain"]
        .preproc.get("VALLEY_NORM_2000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_2000M_SMTHFACT0.5": "sin_valley"})
    )
    dir_valley = np.sin(
        2 * np.pi / 180 * data["terrain"].preproc.get("VALLEY_DIR_2000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_2000M_SMTHFACT0.5": "sin_valley"})
    sin_valley = norm_valley * dir_valley
    sin_valley.attrs.update(data["terrain"].attrs)
    return sin_valley.preproc.interp(stations).astype("float32")


@out_format()
def sin_valley_index_10000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate sine of valley index 10km resolution
    """
    norm_valley = (
        data["terrain"]
        .preproc.get("VALLEY_NORM_10000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_10000M_SMTHFACT0.5": "sin_valley"})
    )
    dir_valley = np.sin(
        2 * np.pi / 180 * data["terrain"].preproc.get("VALLEY_DIR_10000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_10000M_SMTHFACT0.5": "sin_valley"})
    sin_valley = norm_valley * dir_valley
    sin_valley.attrs.update(data["terrain"].attrs)
    return sin_valley.preproc.interp(stations).astype("float32")


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
def sn_derivative_100000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Extract S-N derivative at 100km resolution
    """
    return (
        data["terrain"]
        .preproc.get("SN_DERIVATIVE_100000M_SIGRATIO1")
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
