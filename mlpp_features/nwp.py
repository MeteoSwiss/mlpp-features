import logging
from typing import Dict

import xarray as xr
import numpy as np

from mlpp_features.decorators import asarray

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@asarray
def average_downward_longwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of average downward longwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("surface_downwelling_longwave_flux_in_air")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def boundary_layer_height_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of boundary layer height
    """
    return (
        data["nwp"]
        .preproc.get("atmosphere_boundary_layer_thickness")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def cos_wind_from_direction_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of cosine wind direction
    """
    return (
        data["nwp"]
        .preproc.get(["eastward_wind", "northward_wind"])
        .preproc.wind_from_direction()
        .pipe(lambda x: x * 2 * np.pi / 360)  # radians
        .pipe(np.cos)
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def dew_point_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of dew point temperature
    """
    return (
        data["nwp"]
        .preproc.get("dew_point_temperature")
        .mean("realization")
        .preproc.interp(stations)
        .pipe(lambda x: x - 273.15)  # convert to celsius
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def diffuse_downward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of diffuse downward shortwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("surface_diffuse_downwelling_shortwave_flux_in_air")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def diffuse_upward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of diffuse upward shortwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("surface_upwelling_shortwave_flux_in_air")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def direct_downward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of direct downward shortwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("surface_direct_downwelling_shortwave_flux_in_air")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def eastward_wind_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of average downward longwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("eastward_wind")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def heat_index_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:

    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["air_temperature"]
        data["nwp"]["dew_point_temperature"]
    except KeyError:
        raise KeyError(["air_temperature", "dew_point_temperature"])

    t_celsius = temperature_ensavg(data, stations, reftimes, leadtimes, **kwargs)
    u = relative_humidity_ensavg(data, stations, reftimes, leadtimes, **kwargs)

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

    t_fahrenheit = t_celsius * 1.8 + 32
    hi = xr.where(
        t_fahrenheit > 80,
        _hi_normal_range(t_fahrenheit, u),
        _hi_cold_range(t_fahrenheit, u),
    )

    return hi.astype("float32")


@asarray
def leadtime(data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs):
    """
    Extract leadtime in hours
    """
    if len(data["nwp"]) == 0:
        raise KeyError([])
    ds = data["nwp"]
    return (
        ds.drop_vars(ds.data_vars)
        .drop_dims(("x", "y", "realization"), errors="ignore")
        .preproc.align_time(reftimes, leadtimes)
        .reset_coords("leadtime")
        .astype("float32")
    )


@asarray
def model_height_difference(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Difference between model height and height from the more precise DEM
    """
    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["HSURF"]
        data["terrain"]["DEM"]
    except KeyError:
        raise KeyError(["HSURF", "DEM"])

    hsurf_on_poi = data["nwp"].preproc.get("HSURF").preproc.interp(stations)
    dem_on_poi = data["terrain"].preproc.get("DEM").preproc.interp(stations)
    ds = xr.merge([hsurf_on_poi, dem_on_poi])

    return ds.preproc.difference("HSURF", "DEM").astype("float32")


@asarray
def northward_wind_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of average downward longwave radiation
    """
    return (
        data["nwp"]
        .preproc.get("northward_wind")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def pressure_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of surface pressure
    """
    return (
        data["nwp"]
        .preproc.get("surface_air_pressure")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def pressure_difference_BAS_LUG_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of surface pressure
    """
    ds = data["nwp"].preproc.get("surface_air_pressure")
    station_pair = stations.loc[["BAS", "LUG"]]
    return (
        ds.preproc.interp(station_pair)
        .diff("station")
        .squeeze("station", drop=True)
        .mean("realization")
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def pressure_difference_GVE_GUT_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of surface pressure
    """
    ds_nwp = data["nwp"].preproc.get("surface_air_pressure")
    station_pair = stations.loc[["GVE", "GUT"]]
    return (
        ds_nwp.preproc.get("surface_air_pressure")
        .preproc.interp(station_pair)
        .diff("station")
        .squeeze("station", drop=True)
        .mean("realization")
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def relative_humidity_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of relative humidity
    """
    e = water_vapor_pressure_ensavg(data, stations, reftimes, leadtimes, **kwargs)
    e_s = water_vapor_saturation_pressure_ensavg(
        data, stations, reftimes, leadtimes, **kwargs
    )

    return (e / e_s * 100).astype("float32")


@asarray
def sin_wind_from_direction_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of sine wind direction
    """
    return (
        data["nwp"]
        .preproc.get(["eastward_wind", "northward_wind"])
        .preproc.wind_from_direction()
        .pipe(lambda x: x * 2 * np.pi / 360)  # radians
        .pipe(np.sin)
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def specific_humidity_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of specific humidity
    """
    return (
        data["nwp"]
        .preproc.get("specific_humidity")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def sunshine_duration_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of sunshine duration
    """
    return (
        data["nwp"]
        .preproc.get("duration_of_sunshine")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def sx_500m(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Extract Sx with a 500m radius, for azimuth sectors of 10 degrees
    and every 5 degrees.
    """
    sx = data["terrain"].preproc.get("SX_50M_RADIUS500")
    nsectors = sx.wind_from_direction.size
    degsector = int(360 / nsectors)

    # find correct index for every sample
    wdir = (
        wind_from_direction_ensavg(data, stations, reftimes, leadtimes, **kwargs)
        .astype("int16")
        .load()
    )
    ind = (wdir + degsector / 2) // degsector
    del wdir
    ind = ind.where(ind != nsectors, 0).astype("int8")

    # compute Sx
    station_sub = stations.loc[ind.station]
    sx = sx.preproc.interp(station_sub)
    sx = sx.isel(wind_from_direction=ind.sel(station=sx.station))

    return sx.drop_vars("wind_from_direction")


@asarray
def temperature_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of temperature
    """
    return (
        data["nwp"]
        .preproc.get("air_temperature")
        .mean("realization")
        .preproc.interp(stations)
        .pipe(lambda x: x - 273.15)  # convert to celsius
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def water_vapor_mixing_ratio_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["dew_point_temperature"]
        data["nwp"]["air_temperature"]
        data["nwp"]["surface_air_pressure"]
    except KeyError:
        raise KeyError(
            ["dew_point_temperature", "air_temperature", "surface_air_pressure"]
        )

    try:
        q = specific_humidity_ensavg(data, stations, reftimes, leadtimes, **kwargs)
        return (q / (1 - q)).astype("float32")
    except KeyError:
        e = water_vapor_pressure_ensavg(data, stations, reftimes, leadtimes, **kwargs)
        p = pressure_ensavg(data, stations)
        return ((622.0 * e) / (p / 100 - e)).astype("float32")


@asarray
def water_vapor_pressure_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["dew_point_temperature"]
        data["nwp"]["air_temperature"]
    except KeyError:
        raise KeyError(["dew_point_temperature", "air_temperature"])

    dew_point_temperature = dew_point_ensavg(data, stations, reftimes, leadtimes)
    air_temperature = temperature_ensavg(data, stations, reftimes, leadtimes)

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
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:

    air_temperature = temperature_ensavg(data, stations, reftimes, leadtimes, **kwargs)

    def e_from_t(t, a, b, c):
        return c * np.exp(a * t / (b + t))

    e = xr.where(
        air_temperature > 0,
        e_from_t(air_temperature, 17.368, 238.83, 6.107),
        e_from_t(air_temperature, 17.856, 245.52, 6.108),
    )

    return e.astype("float32")


@asarray
def wind_from_direction_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of wind direction
    """
    return (
        data["nwp"]
        .preproc.get(["eastward_wind", "northward_wind"])
        .preproc.wind_from_direction()
        .preproc.circmean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_speed_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of wind speed
    """
    return (
        data["nwp"]
        .preproc.get(["eastward_wind", "northward_wind"])
        .preproc.norm()
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_speed_ensstd(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble std of wind speed
    """
    return (
        data["nwp"]
        .preproc.get(["eastward_wind", "northward_wind"])
        .preproc.norm()
        .std("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_speed_error(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble mean wind speed
    """
    nwp = wind_speed_ensavg(data, stations, reftimes, leadtimes, **kwargs)
    obs = data["obs"][["wind_speed"]]
    obs = (
        obs.preproc.unstack_time(reftimes, leadtimes)
        .to_array(name="wind_speed")
        .squeeze("variable", drop=True)
    )
    return (nwp - obs).astype("float32")


@asarray
def wind_speed_of_gust_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of wind speed gust
    """
    return (
        data["nwp"]
        .preproc.get("wind_speed_of_gust")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_speed_of_gust_ensstd(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble std of wind speed gust
    """
    return (
        data["nwp"]
        .preproc.get("wind_speed_of_gust")
        .std("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@asarray
def wind_speed_of_gust_error(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble mean wind speed of gust
    """
    nwp = wind_speed_of_gust_ensavg(data, stations, reftimes, leadtimes, **kwargs)
    obs = data["obs"][["wind_speed_of_gust"]]
    obs = (
        obs.preproc.unstack_time(reftimes, leadtimes)
        .to_array(name="wind_speed_of_gust")
        .squeeze("variable", drop=True)
    )
    return (nwp - obs).astype("float32")


@asarray
def wind_gust_factor_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean wind gust factor
    """
    ds_wind = data["nwp"].preproc.get(
        ["eastward_wind", "northward_wind", "wind_speed_of_gust"]
    )
    mean_wind = (ds_wind[["eastward_wind", "northward_wind"]].preproc.norm()).norm
    return (
        (ds_wind.wind_speed_of_gust / mean_wind)
        .to_dataset(name="wind_gust_factor")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )
