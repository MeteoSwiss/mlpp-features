import logging
from typing import Dict

import xarray as xr

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


def average_downward_longwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.Dataset:
    """
    Ensemble mean of average downward longwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("surface_downwelling_longwave_flux_in_air")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def boundary_layer_height_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.Dataset:
    """
    Ensemble mean of boundary layer height
    """
    return (
        data["nwp"]
        .preproc.get("atmosphere_boundary_layer_thickness")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def dew_point_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.Dataset:
    """
    Ensemble mean of dew point temperature
    """
    return (
        data["nwp"]
        .preproc.get("dew_point_temperature")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def diffuse_downward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.Dataset:
    """
    Ensemble mean of diffuse downward shortwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("surface_diffuse_downwelling_shortwave_flux_in_air")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def diffuse_upward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.Dataset:
    """
    Ensemble mean of diffuse upward shortwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("surface_upwelling_shortwave_flux_in_air")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def direct_downward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.Dataset:
    """
    Ensemble mean of direct downward shortwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("surface_direct_downwelling_shortwave_flux_in_air")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def eastward_wind_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.Dataset:
    """
    Ensemble mean of average downward longwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("eastward_wind")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def model_height_difference(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.Dataset:
    """
    Difference between model height and height from the more precise DEM
    """
    hsurf_on_poi = data["nwp"].preproc.get("HSURF").preproc.interp(coords)
    dem_on_poi = data["terrain"].preproc.get("DEM").preproc.interp(coords)
    ds = xr.merge([hsurf_on_poi, dem_on_poi])

    return ds.preproc.difference("HSURF", "DEM").astype("float32")


def northward_wind_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.Dataset:
    """
    Ensemble mean of average downward longwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("northward_wind")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def pressure_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.Dataset:
    """
    Ensemble mean of surface pressure
    """
    return (
        data["nwp"]
        .preproc.get("surface_air_pressure")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def specific_humidity_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.Dataset:
    """
    Ensemble mean of specific humidity
    """
    return (
        data["nwp"]
        .preproc.get("specific_humidity")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def sunshine_duration_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.Dataset:
    """
    Ensemble mean of sunshine duration
    """
    return (
        data["nwp"]
        .preproc.get("duration_of_sunshine")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


def temperature_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.Dataset:
    """
    Ensemble mean of temperature
    """
    return (
        data["nwp"]
        .preproc.get("air_temperature")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )


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


def wind_speed_of_gust_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.Dataset:
    """
    Ensemble mean of wind speed gust
    """
    return (
        data["nwp"]
        .preproc.get(["wind_speed_of_gust"])
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )
