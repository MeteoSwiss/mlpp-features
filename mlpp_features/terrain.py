import logging
from typing import Dict

import numpy as np
import xarray as xr

from mlpp_features.decorators import inputs, out_format
from mlpp_features import geo_calculations as geo


LOGGER = logging.getLogger(__name__)


# Set global options
xr.set_options(keep_attrs=True)


ALPINE_SOUTHERN_CREST_WGS84 = [
    [45.79398, 6.81499],
    [45.81853, 6.96789],
    [45.84573, 7.08259],
    [45.86075, 7.13359],
    [45.86386, 7.18467],
    [45.88483, 7.26551],
    [45.89683, 7.34643],
    [45.91174, 7.43164],
    [45.92364, 7.47427],
    [45.94145, 7.53825],
    [45.95924, 7.58947],
    [45.96214, 7.6364],
    [45.96209, 7.66626],
    [45.94113, 7.72588],
    [45.91406, 7.82377],
    [45.9138, 7.89196],
    [45.90763, 7.94305],
    [45.91049, 7.96865],
    [45.94008, 8.00305],
    [45.98468, 8.00778],
    [46.02034, 8.01669],
    [46.05302, 8.02558],
    [46.13037, 8.02642],
    [46.16611, 8.01824],
    [46.16909, 8.01827],
    [46.22557, 8.02744],
    [46.25811, 8.06211],
    [46.29951, 8.10551],
    [46.31699, 8.16584],
    [46.34322, 8.24782],
    [46.37255, 8.30414],
    [46.414, 8.33061],
    [46.44064, 8.34826],
    [46.47907, 8.37906],
    [46.50552, 8.4183],
    [46.52608, 8.44885],
    [46.55771, 8.5659],
    [46.56881, 8.64379],
    [46.57401, 8.71294],
    [46.55536, 8.78151],
    [46.51602, 8.83657],
    [46.49983, 8.93957],
    [46.48669, 9.02968],
    [46.48235, 9.12433],
    [46.47834, 9.19312],
    [46.48306, 9.27082],
    [46.49433, 9.30999],
    [46.4968, 9.34024],
    [46.49012, 9.38309],
    [46.46893, 9.40387],
    [46.43279, 9.42838],
    [46.41449, 9.45353],
    [46.39305, 9.48714],
    [46.38029, 9.53397],
    [46.37941, 9.58124],
    [46.38206, 9.59854],
    [46.39627, 9.63352],
    [46.38871, 9.71494],
    [46.37935, 9.73605],
    [46.36036, 9.79114],
    [46.35956, 9.8298],
    [46.38176, 9.90475],
    [46.39847, 9.95714],
    [46.4307, 9.98016],
    [46.48642, 10.01725],
    [46.51289, 10.03145],
    [46.54837, 10.04178],
    [46.55997, 10.05528],
    [46.57147, 10.0731],
    [46.59793, 10.08735],
    [46.61516, 10.11411],
    [46.66797, 10.147],
    [46.69677, 10.18739],
    [46.71059, 10.23137],
    [46.71757, 10.30965],
    [46.72624, 10.43566],
    [46.72144, 10.6129],
    [46.7216, 10.71249],
    [46.71793, 10.83782],
    [46.71559, 10.9156],
]


@out_format()
def aspect_500m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain aspect at a 500m scale
    """
    return (
        data["terrain"]
        .mlpp.get("ASPECT_500M_SIGRATIO1")
        .mlpp.interp(stations)
        .astype("float32")
    )


@out_format()
def aspect_2000m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain aspect at a 2000m scale
    """
    return (
        data["terrain"]
        .mlpp.get("ASPECT_2000M_SIGRATIO1")
        .mlpp.interp(stations)
        .astype("float32")
    )


@inputs("terrain:VALLEY_NORM_1000M_SMTHFACT0.5", "terrain:VALLEY_DIR_1000M_SMTHFACT0.5")
@out_format()
def cos_valley_index_1000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate cosine of valley index 1km resolution
    """
    norm_valley = (
        data["terrain"]
        .mlpp.get("VALLEY_NORM_1000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_1000M_SMTHFACT0.5": "cos_valley"})
    )
    dir_valley = np.cos(
        2 * np.pi / 180 * data["terrain"].mlpp.get("VALLEY_DIR_1000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_1000M_SMTHFACT0.5": "cos_valley"})
    cos_valley = norm_valley * dir_valley
    cos_valley.attrs.update(data["terrain"].attrs)
    return cos_valley.mlpp.interp(stations).astype("float32")


@inputs("terrain:VALLEY_NORM_2000M_SMTHFACT0.5", "terrain:VALLEY_DIR_2000M_SMTHFACT0.5")
@out_format()
def cos_valley_index_2000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate cosine of valley index 2km resolution
    """
    norm_valley = (
        data["terrain"]
        .mlpp.get("VALLEY_NORM_2000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_2000M_SMTHFACT0.5": "cos_valley"})
    )
    dir_valley = np.cos(
        2 * np.pi / 180 * data["terrain"].mlpp.get("VALLEY_DIR_2000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_2000M_SMTHFACT0.5": "cos_valley"})
    cos_valley = norm_valley * dir_valley
    cos_valley.attrs.update(data["terrain"].attrs)
    return cos_valley.mlpp.interp(stations).astype("float32")


@inputs(
    "terrain:VALLEY_NORM_10000M_SMTHFACT0.5", "terrain:VALLEY_DIR_10000M_SMTHFACT0.5"
)
@out_format()
def cos_valley_index_10000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate cosine of valley index 10km resolution
    """
    norm_valley = (
        data["terrain"]
        .mlpp.get("VALLEY_NORM_10000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_10000M_SMTHFACT0.5": "cos_valley"})
    )
    dir_valley = np.cos(
        2 * np.pi / 180 * data["terrain"].mlpp.get("VALLEY_DIR_10000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_10000M_SMTHFACT0.5": "cos_valley"})
    cos_valley = norm_valley * dir_valley
    cos_valley.attrs.update(data["terrain"].attrs)
    return cos_valley.mlpp.interp(stations).astype("float32")


@inputs()
@out_format()
def distance_to_alpine_ridge(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Compute horizontal distance to the main Alpine ridge
    """
    points = [(lat, lon) for lat, lon in zip(stations.latitude, stations.longitude)]
    distances = geo.distances_points_to_line(points, ALPINE_SOUTHERN_CREST_WGS84)
    return xr.Dataset(
        coords={
            "station": stations.index,
            "longitude": ("station", stations.longitude),
            "latitude": ("station", stations.latitude),
            "height_masl": ("station", stations.height_masl),
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
    return data["terrain"].mlpp.get("DEM").mlpp.interp(stations).astype("float32")


@inputs("terrain:VALLEY_NORM_1000M_SMTHFACT0.5", "terrain:VALLEY_DIR_1000M_SMTHFACT0.5")
@out_format()
def sin_valley_index_1000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate sine of valley index 1km resolution
    """
    norm_valley = (
        data["terrain"]
        .mlpp.get("VALLEY_NORM_1000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_1000M_SMTHFACT0.5": "sin_valley"})
    )
    dir_valley = np.sin(
        2 * np.pi / 180 * data["terrain"].mlpp.get("VALLEY_DIR_1000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_1000M_SMTHFACT0.5": "sin_valley"})
    sin_valley = norm_valley * dir_valley
    sin_valley.attrs.update(data["terrain"].attrs)
    return sin_valley.mlpp.interp(stations).astype("float32")


@inputs("terrain:VALLEY_NORM_2000M_SMTHFACT0.5", "terrain:VALLEY_DIR_2000M_SMTHFACT0.5")
@out_format()
def sin_valley_index_2000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate sine of valley index 2km resolution
    """
    norm_valley = (
        data["terrain"]
        .mlpp.get("VALLEY_NORM_2000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_2000M_SMTHFACT0.5": "sin_valley"})
    )
    dir_valley = np.sin(
        2 * np.pi / 180 * data["terrain"].mlpp.get("VALLEY_DIR_2000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_2000M_SMTHFACT0.5": "sin_valley"})
    sin_valley = norm_valley * dir_valley
    sin_valley.attrs.update(data["terrain"].attrs)
    return sin_valley.mlpp.interp(stations).astype("float32")


@inputs(
    "terrain:VALLEY_NORM_10000M_SMTHFACT0.5", "terrain:VALLEY_DIR_10000M_SMTHFACT0.5"
)
@out_format()
def sin_valley_index_10000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Calculate sine of valley index 10km resolution
    """
    norm_valley = (
        data["terrain"]
        .mlpp.get("VALLEY_NORM_10000M_SMTHFACT0.5")
        .rename({"VALLEY_NORM_10000M_SMTHFACT0.5": "sin_valley"})
    )
    dir_valley = np.sin(
        2 * np.pi / 180 * data["terrain"].mlpp.get("VALLEY_DIR_10000M_SMTHFACT0.5")
    ).rename({"VALLEY_DIR_10000M_SMTHFACT0.5": "sin_valley"})
    sin_valley = norm_valley * dir_valley
    sin_valley.attrs.update(data["terrain"].attrs)
    return sin_valley.mlpp.interp(stations).astype("float32")


@out_format()
def slope_500m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Extract slope at 500m resolution
    """
    return (
        data["terrain"]
        .mlpp.get("SLOPE_500M_SIGRATIO1")
        .mlpp.interp(stations)
        .astype("float32")
    )


@out_format()
def slope_2000m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Extract slope at 2000m resolution
    """
    return (
        data["terrain"]
        .mlpp.get("SLOPE_2000M_SIGRATIO1")
        .mlpp.interp(stations)
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
        .mlpp.get("SN_DERIVATIVE_500M_SIGRATIO1")
        .mlpp.interp(stations)
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
        .mlpp.get("SN_DERIVATIVE_2000M_SIGRATIO1")
        .mlpp.interp(stations)
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
        .mlpp.get("SN_DERIVATIVE_100000M_SIGRATIO1")
        .mlpp.interp(stations)
        .astype("float32")
    )


@out_format()
def std_500m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain STD at a 500m scale
    """
    return data["terrain"].mlpp.get("STD_500M").mlpp.interp(stations).astype("float32")


@out_format()
def std_2000m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain STD at a 2000m scale
    """
    return data["terrain"].mlpp.get("STD_2000M").mlpp.interp(stations).astype("float32")


@out_format()
def tpi_500m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain TPI at a 500m scale
    """
    return data["terrain"].mlpp.get("TPI_500M").mlpp.interp(stations).astype("float32")


@out_format()
def tpi_2000m(data: Dict[str, xr.Dataset], stations, *args, **kwargs) -> xr.Dataset:
    """
    Terrain TPI at a 2000m scale
    """
    return data["terrain"].mlpp.get("TPI_2000M").mlpp.interp(stations).astype("float32")


@out_format()
def valley_norm_2000m(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Extract valley norm 2km resolution
    """
    return (
        data["terrain"]
        .mlpp.get("VALLEY_NORM_2000M_SMTHFACT0.5")
        .mlpp.interp(stations)
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
        .mlpp.get("VALLEY_NORM_20000M_SMTHFACT0.5")
        .mlpp.interp(stations)
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
        .mlpp.get("WE_DERIVATIVE_500M_SIGRATIO1")
        .mlpp.interp(stations)
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
        .mlpp.get("WE_DERIVATIVE_2000M_SIGRATIO1")
        .mlpp.interp(stations)
        .astype("float32")
    )
