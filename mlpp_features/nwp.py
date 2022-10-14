import logging
from typing import Dict

import xarray as xr
import numpy as np

from mlpp_features.decorators import asarray, reuse
from mlpp_features import calc

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@reuse
@asarray
def air_temperature(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble of temperature in °C
    """
    return (
        data["nwp"]
        .preproc.get("air_temperature")
        .preproc.interp(stations)
        .pipe(lambda x: x - 273.15)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def air_temperature_ensavg(
    data, stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of temperature in °C
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return t.mean("realization")


@reuse
@asarray
def air_temperature_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run temperature in °C
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return t.isel(realization=0, drop=True)


@reuse
@asarray
def air_temperature_ensstd(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble standard deviation of temperature in °C
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return t.std("realization")


@reuse
@asarray
def air_temperature_dailyrange(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of daily temperature range in °C
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    daymax = t.preproc.daystat(xr.Dataset.max)
    daymin = t.preproc.daystat(xr.Dataset.min)
    return daymax - daymin


@reuse
@asarray
def air_temperature_dailyrange_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of daily temperature range in °C
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return t.mean("realization")


@reuse
@asarray
def average_downward_longwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of average downward longwave radiation in W m-2
    """
    return (
        data["nwp"]
        .preproc.get("surface_downwelling_longwave_flux_in_air")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def boundary_layer_height(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble of boundary layer height in m
    """
    return (
        data["nwp"]
        .preproc.get("atmosphere_boundary_layer_thickness")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def boundary_layer_height_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of boundary layer height in m
    """
    blh = boundary_layer_height(data, stations, reftimes, leadtimes, **kwargs)
    return blh.mean("realization")


@reuse
@asarray
def boundary_layer_height_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of boundary layer height in m
    """
    blh = boundary_layer_height(data, stations, reftimes, leadtimes, **kwargs)
    return blh.isel(realization=0, drop=True)


@reuse
@asarray
def cos_wind_from_direction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of cosine wind direction
    """
    wdir = wind_from_direction(data, stations, reftimes, leadtimes, **kwargs)
    return np.cos(wdir * 2 * np.pi / 360)


@reuse
@asarray
def cos_wind_from_direction_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of cosine wind direction
    """
    wdir = cos_wind_from_direction(data, stations, reftimes, leadtimes, **kwargs)
    return wdir.mean("realization")


@reuse
@asarray
def cos_wind_from_direction_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble control of cosine wind direction
    """
    wdir = cos_wind_from_direction(data, stations, reftimes, leadtimes, **kwargs)
    return wdir.isel(realization=0, drop=True)


@reuse
@asarray
def dew_point_depression(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of dew point depression (T - T_d)
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    t_d = dew_point_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return (t - t_d).astype("float32")


@reuse
@asarray
def dew_point_depression_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of dew point depression (T - T_d)
    """
    tdep = dew_point_depression(data, stations, reftimes, leadtimes, **kwargs)
    return tdep.mean("realization")


@reuse
@asarray
def dew_point_depression_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run dew point depression (T - T_d)
    """
    tdep = dew_point_depression(data, stations, reftimes, leadtimes, **kwargs)
    return tdep.isel(realization=0, drop=True)


@reuse
@asarray
def dew_point_temperature(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble of dew point temperature in °C
    """
    return (
        data["nwp"]
        .preproc.get("dew_point_temperature")
        .preproc.interp(stations)
        .pipe(lambda x: x - 273.15)  # convert to celsius
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def dew_point_temperature_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of dew point temperature in °C
    """
    td = dew_point_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return td.mean("realization")


@reuse
@asarray
def dew_point_temperature_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run dew point temperature in °C
    """
    td = dew_point_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return td.isel(realization=0, drop=True)


@reuse
@asarray
def equivalent_potential_temperature(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run equivalent potential temperature in °C
    """
    try:
        data["nwp"]["air_temperature"]
        data["nwp"]["dew_point_temperature"]
        data["nwp"]["surface_air_pressure"]
    except KeyError:
        raise KeyError(
            ["air_temperature", "dew_point_temperature", "surface_air_pressure"]
        )

    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    rh = relative_humidity(data, stations, reftimes, leadtimes, **kwargs)
    p = surface_air_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return calc.equivalent_potential_temperature_from_t_rh_p(t, rh, p)


@reuse
@asarray
def equivalent_potential_temperature_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run equivalent potential temperature in °C
    """
    theta_e = equivalent_potential_temperature(
        data, stations, reftimes, leadtimes, **kwargs
    )
    return theta_e.mean("realization")


@reuse
@asarray
def equivalent_potential_temperature_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run equivalent potential temperature in °C
    """
    theta_e = equivalent_potential_temperature(
        data, stations, reftimes, leadtimes, **kwargs
    )
    return theta_e.isel(realization=0, drop=True)


@reuse
@asarray
def diffuse_downward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of diffuse downward shortwave radiation in W/m^2
    """
    return (
        data["nwp"]
        .preproc.get("surface_diffuse_downwelling_shortwave_flux_in_air")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def diffuse_upward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of diffuse upward shortwave radiation in W/m^2
    """
    return (
        data["nwp"]
        .preproc.get("surface_upwelling_shortwave_flux_in_air")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def direct_downward_shortwave_radiation_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of direct downward shortwave radiation in W/m^2
    """
    return (
        data["nwp"]
        .preproc.get("surface_direct_downwelling_shortwave_flux_in_air")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


def eastward_wind(data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs):
    return (
        data["nwp"]
        .preproc.get("eastward_wind")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def eastward_wind_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of eastward wind in m/s
    """
    u = eastward_wind(data, stations, reftimes, leadtimes, **kwargs)
    return u.mean("realization")


@reuse
@asarray
def eastward_wind_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of eastward wind in m/s
    """
    u = eastward_wind(data, stations, reftimes, leadtimes, **kwargs)
    return u.isel(realization=0, drop=True)


@reuse
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


@reuse
@asarray
def model_height_difference(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Difference between model height and height from the more precise DEM in m
    """
    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["HSURF"]
        data["terrain"]["DEM"]
    except KeyError:
        raise KeyError(["HSURF", "DEM"])

    hsurf_on_poi = data["nwp"].preproc.get("HSURF").preproc.interp(stations)
    dem_on_poi = data["terrain"].preproc.get("DEM").preproc.interp(stations)

    # drop grid coordinates to avoid conflicts when merging
    hsurf_on_poi = hsurf_on_poi.drop_vars(("x", "y"), errors="ignore")
    dem_on_poi = dem_on_poi.drop_vars(("x", "y"), errors="ignore")

    ds = xr.merge([hsurf_on_poi, dem_on_poi])

    return ds.preproc.difference("HSURF", "DEM").astype("float32")


def northward_wind(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
):
    return (
        data["nwp"]
        .preproc.get("northward_wind")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def northward_wind_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of northward wind in m/s
    """
    v = northward_wind(data, stations, reftimes, leadtimes, **kwargs)
    return v.mean("realization")


@reuse
@asarray
def northward_wind_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of northward wind in m/s
    """
    v = northward_wind(data, stations, reftimes, leadtimes, **kwargs)
    return v.isel(realization=0, drop=True)


@asarray
def potential_temperature(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of potential temperature in °C
    """
    try:
        data["nwp"]["air_temperature"]
        data["nwp"]["surface_air_pressure"]
    except KeyError:
        raise KeyError(["air_temperature", "surface_air_pressure"])

    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    p = surface_air_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return calc.potential_temperature_from_t_and_p(t, p)


@asarray
def potential_temperature_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of potential temperature in °C
    """
    v = potential_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return v.mean("realization")


@asarray
def potential_temperature_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of potential temperature in °C
    """
    v = potential_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return v.isel(realization=0, drop=True)


@reuse
@asarray
def pressure_difference_BAS_LUG(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of pressure difference between Basel and Lugano in hPa
    """
    p = surface_air_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return p.sel(station=["BAS", "LUG"]).diff("station").squeeze("station", drop=True)


@reuse
@asarray
def pressure_difference_BAS_LUG_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of pressure difference between Geneva and Güttingen in hPa
    """
    pdiff = pressure_difference_BAS_LUG(data, stations, reftimes, leadtimes, **kwargs)
    return pdiff.mean("realization")


@reuse
@asarray
def pressure_difference_BAS_LUG_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of pressure difference between Basel and Lugano in hPa
    """
    pdiff = pressure_difference_BAS_LUG(data, stations, reftimes, leadtimes, **kwargs)
    return pdiff.isel(realization=0, drop=True)


@reuse
@asarray
def pressure_difference_GVE_GUT(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of pressure difference between Geneva and Güttingen in hPa
    """
    p = surface_air_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return p.sel(station=["GVE", "GUT"]).diff("station").squeeze("station", drop=True)


@reuse
@asarray
def pressure_difference_GVE_GUT_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of pressure difference between Geneva and Güttingen in hPa
    """
    pdiff = pressure_difference_GVE_GUT(data, stations, reftimes, leadtimes, **kwargs)
    return pdiff.mean("realization")


@reuse
@asarray
def relative_humidity(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of relative humidity in %
    """
    try:
        data["nwp"]["air_temperature"]
        data["nwp"]["dew_point_temperature"]
    except KeyError:
        raise KeyError(["air_temperature", "dew_point_temperature"])

    e = water_vapor_pressure(data, stations, reftimes, leadtimes, **kwargs)
    e_s = water_vapor_saturation_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return (e / e_s * 100).astype("float32")


@reuse
@asarray
def relative_humidity_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of relative humidity in %
    """
    rh = relative_humidity(data, stations, reftimes, leadtimes, **kwargs)
    return rh.mean("realization")


@reuse
@asarray
def relative_humidity_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run relative humidity in %
    """
    rh = relative_humidity(data, stations, reftimes, leadtimes, **kwargs)
    return rh.mean("realization")


@reuse
@asarray
def sin_wind_from_direction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of sine wind direction
    """
    wdir = wind_from_direction(data, stations, reftimes, leadtimes, **kwargs)
    return np.sin(wdir * 2 * np.pi / 360)


@reuse
@asarray
def sin_wind_from_direction_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of sine wind direction
    """
    wdir = sin_wind_from_direction(data, stations, reftimes, leadtimes, **kwargs)
    return wdir.mean("realization")


@reuse
@asarray
def sin_wind_from_direction_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble control of sine wind direction
    """
    wdir = sin_wind_from_direction(data, stations, reftimes, leadtimes, **kwargs)
    return wdir.isel(realization=0, drop=True)


@reuse
@asarray
def specific_humidity(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of specific humidity in g/kg
    """
    return (
        data["nwp"]
        .preproc.get("specific_humidity")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def specific_humidity_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of specific humidity in g/kg
    """
    q = specific_humidity(data, stations, reftimes, leadtimes, **kwargs)
    return q.mean("realization")


@reuse
@asarray
def specific_humidity_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run specific humidity in g/kg
    """
    q = specific_humidity(data, stations, reftimes, leadtimes, **kwargs)
    return q.isel(realization=0, drop=True)


@reuse
@asarray
def sunshine_duration_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of sunshine duration in seconds
    """
    return (
        data["nwp"]
        .preproc.get("duration_of_sunshine")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def surface_air_pressure(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of surface pressure in hPa
    """
    return (
        data["nwp"]
        .preproc.get("surface_air_pressure")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .pipe(lambda x: x / 100)  # Pa to hPa
        .astype("float32")
    )


@reuse
@asarray
def surface_air_pressure_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of surface pressure in hPa
    """
    q = surface_air_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return q.mean("realization")


@reuse
@asarray
def surface_air_pressure_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run surface pressure in hPa
    """
    q = surface_air_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return q.isel(realization=0, drop=True)


@reuse
@asarray
def sx_500m(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Extract ensemble average Sx with a 500m radius, for azimuth sectors of 10 degrees
    and every 5 degrees.
    """
    sx = data["terrain"].preproc.get("SX_50M_RADIUS500")
    nsectors = sx.wind_from_direction.size
    degsector = int(360 / nsectors)

    # find correct index for every sample
    wdir = wind_from_direction(data, stations, reftimes, leadtimes, **kwargs)
    wdir = wdir.astype("int16").load()

    ind = (wdir + degsector / 2) // degsector
    del wdir
    ind = ind.where(ind != nsectors, 0).astype("int8")

    # compute Sx
    station_sub = stations.loc[ind.station]
    sx = sx.preproc.interp(station_sub)
    sx = sx.isel(wind_from_direction=ind.sel(station=sx.station))

    return sx.drop_vars("wind_from_direction")


@reuse
@asarray
def sx_500m_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Extract ensemble average Sx with a 500m radius, for azimuth sectors of 10 degrees
    and every 5 degrees.
    """
    sx = sx_500m(data, stations, reftimes, leadtimes, **kwargs)
    return sx.mean("realization")


@reuse
@asarray
def sx_500m_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Extract ensemble control Sx with a 500m radius, for azimuth sectors of 10 degrees
    and every 5 degrees.
    """
    sx = sx_500m(data, stations, reftimes, leadtimes, **kwargs)
    return sx.isel(realization=0, drop=True)


@reuse
@asarray
def water_vapor_mixing_ratio(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor mixing ratio in g/kg
    """
    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["dew_point_temperature"]
        data["nwp"]["air_temperature"]
        data["nwp"]["surface_air_pressure"]
    except KeyError:
        raise KeyError(
            ["dew_point_temperature", "air_temperature", "surface_air_pressure"]
        )

    e = water_vapor_pressure(data, stations, reftimes, leadtimes, **kwargs)
    p = surface_air_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return calc.mixing_ratio_from_p_and_e(p, e)


@reuse
@asarray
def water_vapor_mixing_ratio_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor mixing ratio in g/kg
    """
    r = water_vapor_mixing_ratio(data, stations, reftimes, leadtimes, **kwargs)
    return r.mean("realization")


@reuse
@asarray
def water_vapor_mixing_ratio_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run water vapor mixing ratio in g/kg
    """
    r = water_vapor_mixing_ratio(data, stations, reftimes, leadtimes, **kwargs)
    return r.isel(realization=0, drop=True)


@reuse
@asarray
def water_vapor_pressure(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor partial pressure
    """
    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["dew_point_temperature"]
        data["nwp"]["air_temperature"]
    except KeyError:
        raise KeyError(["dew_point_temperature", "air_temperature"])

    t_d = dew_point_temperature(data, stations, reftimes, leadtimes, **kwargs)
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return calc.water_vapor_pressure_from_t_and_td(t, t_d)


@reuse
@asarray
def water_vapor_pressure_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor partial pressure
    """
    e = water_vapor_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return e.mean("realization")


@reuse
@asarray
def water_vapor_pressure_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run water vapor partial pressure
    """
    e = water_vapor_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return e.isel(realization=0, drop=True)


@reuse
@asarray
def water_vapor_saturation_pressure(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor partial pressure at saturation in hPa
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return calc.water_vapor_saturation_pressure_from_t(t)


@reuse
@asarray
def water_vapor_saturation_pressure_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor partial pressure at saturation in hPa
    """
    e_s = water_vapor_saturation_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return e_s.mean("realization")


@reuse
@asarray
def water_vapor_saturation_pressure_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run water vapor partial pressure at saturation
    """
    e_s = water_vapor_saturation_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return e_s.isel(realization=0, drop=True)


def wind_speed(data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs):
    u = eastward_wind(data, stations, reftimes, leadtimes, **kwargs)
    v = northward_wind(data, stations, reftimes, leadtimes, **kwargs)
    return np.sqrt(u**2 + v**2)


def wind_from_direction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble control of wind direction
    """
    u = eastward_wind(data, stations, reftimes, leadtimes, **kwargs)
    v = northward_wind(data, stations, reftimes, leadtimes, **kwargs)
    out = (270 - 180 / np.pi * np.arctan2(v, u)) % 360
    out.attrs["units"] = "degrees"
    return out


@reuse
@asarray
def wind_from_direction_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of wind direction
    """
    d = wind_from_direction(data, stations, reftimes, leadtimes, **kwargs)
    return d.mean("realization")


@reuse
@asarray
def wind_from_direction_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble control of wind direction
    """
    d = wind_from_direction(data, stations, reftimes, leadtimes, **kwargs)
    return d.isel(realization=0, drop=True)


@reuse
@asarray
def wind_speed_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of wind speed
    """
    uv = wind_speed(data, stations, reftimes, leadtimes, **kwargs)
    return uv.mean("realization")


@reuse
@asarray
def wind_speed_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of wind speed
    """
    uv = wind_speed(data, stations, reftimes, leadtimes, **kwargs)
    return uv.isel(realization=0, drop=True)


@reuse
@asarray
def wind_speed_ensstd(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble std of wind speed
    """
    uv = wind_speed(data, stations, reftimes, leadtimes, **kwargs)
    return uv.std("realization")


@reuse
@asarray
def wind_speed_error(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble mean wind speed
    """
    nwp = wind_speed(data, stations, reftimes, leadtimes, **kwargs)
    obs = data["obs"][["wind_speed"]]
    obs = (
        obs.preproc.unstack_time(reftimes, leadtimes)
        .to_array(name="wind_speed")
        .squeeze("variable", drop=True)
    )
    return (nwp - obs).astype("float32")


@reuse
@asarray
def wind_speed_error_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble mean wind speed
    """
    uv = wind_speed_error(data, stations, reftimes, leadtimes, **kwargs)
    return uv.mean("realization")


@reuse
@asarray
def wind_speed_error_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble control wind speed
    """
    uv = wind_speed_error(data, stations, reftimes, leadtimes, **kwargs)
    return uv.isel(realization=0, drop=True)


@reuse
@asarray
def wind_speed_of_gust(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble of wind speed gust
    """
    return (
        data["nwp"]
        .preproc.get("wind_speed_of_gust")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def wind_speed_of_gust_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of wind speed gust
    """
    ug = wind_speed_of_gust(data, stations, reftimes, leadtimes, **kwargs)
    return ug.mean("realization")


@reuse
@asarray
def wind_speed_of_gust_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of wind speed of gust
    """
    ug = wind_speed_of_gust(data, stations, reftimes, leadtimes, **kwargs)
    return ug.isel(realization=0, drop=True)


@reuse
@asarray
def wind_speed_of_gust_ensstd(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble std of wind speed gust
    """
    ug = wind_speed_of_gust(data, stations, reftimes, leadtimes, **kwargs)
    return ug.std("realization")


@reuse
@asarray
def wind_speed_of_gust_error_ensavg(
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


@reuse
@asarray
def wind_speed_of_gust_error_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble control wind speed of gust
    """
    nwp = wind_speed_of_gust_ensctrl(data, stations, reftimes, leadtimes, **kwargs)
    obs = data["obs"][["wind_speed_of_gust"]]
    obs = (
        obs.preproc.unstack_time(reftimes, leadtimes)
        .to_array(name="wind_speed_of_gust")
        .squeeze("variable", drop=True)
    )
    return (nwp - obs).astype("float32")


@reuse
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
        ((ds_wind.wind_speed_of_gust + 1.0) / (mean_wind + 1.0))
        .to_dataset(name="wind_gust_factor")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@reuse
@asarray
def wind_gust_factor_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble control of wind gust factor
    """
    ds_wind = data["nwp"].preproc.get(
        ["eastward_wind", "northward_wind", "wind_speed_of_gust"]
    )
    mean_wind = (
        ds_wind[["eastward_wind", "northward_wind"]]
        .isel(realization=0, drop=True)
        .preproc.norm()
    )
    gust_wind = ds_wind[["wind_speed_of_gust"]].isel(realization=0, drop=True)
    return (
        ((gust_wind.wind_speed_of_gust + 1.0) / (mean_wind.norm + 1.0))
        .to_dataset(name="wind_gust_factor")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )
