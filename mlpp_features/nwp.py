import logging
from typing import Dict

import xarray as xr
import numpy as np

from mlpp_features.utils import asarray

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@asarray
def average_downward_longwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
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


@asarray
def boundary_layer_height_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
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


@asarray
def dew_point_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.DataArray:
    """
    Ensemble mean of dew point temperature
    """
    return (
        data["nwp"]
        .preproc.get("dew_point_temperature")
        .mean("realization")
        .preproc.interp(coords)
        .preproc.apply_func(lambda x: x - 273.15)  # convert to celsius
        .astype("float32")
    )


@asarray
def diffuse_downward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
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


@asarray
def diffuse_upward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
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


@asarray
def direct_downward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
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


@asarray
def eastward_wind_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.DataArray:
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


@asarray
def heat_index_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.DataArray:

    t_fahrenheit = temperature_ensavg(data, coords) * 1.8 + 32
    u = relative_humidity_ensavg(data, coords)

    def _hi_normal_range(t_f, u):
        return (
            -42.379
            + 2.04901523 * t_f
            + 10.14333127 * u
            - 0.22475541 * t_f * u
            - 6.83783e-3 * t_f ** 2
            - 5.481717e-2 * u ** 2
            + 1.22874e-3 * (t_f ** 2) * u
            + 8.5282e-4 * t_f * (u ** 2)
            - 1.99e-6 * (t_f ** 2) * (u ** 2)
        )

    def _hi_cold_range(t_f, u):
        return 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (u * 0.094))

    hi = xr.where(
        t_fahrenheit > 80,
        _hi_normal_range(t_fahrenheit, u),
        _hi_cold_range(t_fahrenheit, u),
    )

    return hi.astype("float32")


@asarray
def model_height_difference(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
    """
    Difference between model height and height from the more precise DEM
    """
    hsurf_on_poi = data["nwp"].preproc.get("HSURF").preproc.interp(coords)
    dem_on_poi = data["terrain"].preproc.get("DEM").preproc.interp(coords)
    ds = xr.merge([hsurf_on_poi, dem_on_poi])

    return ds.preproc.difference("HSURF", "DEM").astype("float32")


@asarray
def northward_wind_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
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


@asarray
def pressure_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.DataArray:
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


@asarray
def relative_humidity_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of relative humidity
    """

    e = water_vapor_pressure_ensavg(data, coords)
    e_s = water_vapor_saturation_pressure_ensavg(data, coords)

    return (e / e_s * 100).astype("float32")


@asarray
def specific_humidity_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
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


@asarray
def sunshine_duration_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
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


@asarray
def temperature_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.DataArray:
    """
    Ensemble mean of temperature
    """
    return (
        data["nwp"]
        .preproc.get("air_temperature")
        .mean("realization")
        .preproc.interp(coords)
        .preproc.apply_func(lambda x: x - 273.15)  # convert to celsius
        .astype("float32")
    )


@asarray
def water_vapor_mixing_ratio_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:

    try:
        q = specific_humidity_ensavg(data, coords)
        return (q / (1 - q)).astype("float32")
    except KeyError:
        e = water_vapor_pressure_ensavg(data, coords)
        p = pressure_ensavg(data, coords)
        return ((622.0 * e) / (p / 100 - e)).astype("float32")


@asarray
def water_vapor_pressure_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:

    dew_point_temperature = dew_point_ensavg(data, coords)
    air_temperature = temperature_ensavg(data, coords)

    def e_from_t(t, a, b, c):
        return c * np.exp(a * t / (b + t))

    e = xr.where(
        air_temperature > 0,
        e_from_t(dew_point_temperature, 17.368, 238.83, 6.107),
        e_from_t(dew_point_temperature, 17.856, 245.52, 6.108),
    )

    return e.astype("float32")


@asarray
def water_vapor_saturation_pressure_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:

    air_temperature = temperature_ensavg(data, coords)

    def e_from_t(t, a, b, c):
        return c * np.exp(a * t / (b + t))

    e = xr.where(
        air_temperature > 0,
        e_from_t(air_temperature, 17.368, 238.83, 6.107),
        e_from_t(air_temperature, 17.856, 245.52, 6.108),
    )

    return e.astype("float32")


@asarray
def wind_speed_ensavg(data: Dict[str, xr.Dataset], coords, **kwargs) -> xr.DataArray:
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


@asarray
def wind_speed_of_gust_ensavg(
    data: Dict[str, xr.Dataset], coords, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of wind speed gust
    """
    return (
        data["nwp"]
        .preproc.get("wind_speed_of_gust")
        .mean("realization")
        .preproc.interp(coords)
        .astype("float32")
    )
