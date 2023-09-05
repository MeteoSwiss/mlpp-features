import logging
from typing import Dict

import xarray as xr
import numpy as np

from mlpp_features.decorators import cache, out_format
from mlpp_features import calc

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@cache
def _air_temperature_ens(data: Dict[str, xr.Dataset], stations, **kwargs) -> xr.Dataset:
    """
    Ensemble of temperature in °C
    """
    return (
        data["nwp"]
        .preproc.get("air_temperature")
        .preproc.interp(stations, **kwargs)
        .pipe(lambda x: x - 273.15)
        .astype("float32")
    )


@out_format(units="degC")
def air_temperature_ens(
    data: Dict[str, xr.Dataset], stations, reftimes=None, leadtimes=None, **kwargs
) -> xr.DataArray:
    """
    Ensemble of temperature in °C
    """
    ens_data = _air_temperature_ens(data, stations, **kwargs)
    ens_data = ens_data.preproc.align_time(reftimes, leadtimes)
    return ens_data


@out_format(units="degC")
def air_temperature_ensavg(
    data, stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of temperature in °C
    """
    t = air_temperature_ens(data, stations, **kwargs)
    return t.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="degC")
def air_temperature_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run temperature in °C
    """
    t = air_temperature_ens(data, stations, **kwargs)
    return (
        t.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="degC")
def air_temperature_ensstd(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble standard deviation of temperature in °C
    """
    t = air_temperature_ens(data, stations, **kwargs)
    return t.std("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="degC")
def air_temperature_dailyrange_ens(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of daily temperature range in °C
    """
    t = air_temperature_ens(data, stations, *args, **kwargs).to_dataset()
    t = t.assign_coords(time=t.forecast_reference_time + t.t)
    daymax = t.preproc.daystat(xr.Dataset.max)
    daymin = t.preproc.daystat(xr.Dataset.min)
    return daymax - daymin


@out_format(units="degC")
def air_temperature_dailyrange_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of daily temperature range in °C
    """
    t = air_temperature_ens(data, stations, **kwargs)
    return t.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="W m-2")
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


@out_format(units="m")
def boundary_layer_height_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of boundary layer height in m
    """
    return (
        data["nwp"]
        .preproc.get("atmosphere_boundary_layer_thickness")
        .mean("realization")
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m")
def boundary_layer_height_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of boundary layer height in m
    """
    return (
        data["nwp"]
        .preproc.get("atmosphere_boundary_layer_thickness")
        .isel(realization=0, drop=True)
        .preproc.interp(stations)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format()
def cos_wind_from_direction_ens(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble of cosine wind direction
    """
    wdir = wind_from_direction_ens(data, stations, reftimes, leadtimes, **kwargs)
    return np.cos(wdir * 2 * np.pi / 360)


@out_format()
def cos_wind_from_direction_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of cosine wind direction
    """
    wdir = cos_wind_from_direction_ens(data, stations, reftimes, leadtimes, **kwargs)
    return wdir.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format()
def cos_wind_from_direction_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble control of cosine wind direction
    """
    wdir = cos_wind_from_direction_ens(data, stations, reftimes, leadtimes, **kwargs)
    return (
        wdir.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="degrees")
def cos_wind_from_direction_ensavg_error(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble average cosine wind direction
    """
    nwp = cos_wind_from_direction_ensavg(data, stations, reftimes, leadtimes, **kwargs)
    obs = (
        data["obs"]
        .preproc.get("wind_from_direction")
        .pipe(lambda x: x * 2 * np.pi / 360)  # to radians
        .pipe(np.cos)
        .preproc.unstack_time(reftimes, leadtimes)
        .to_array(name="wind_speed")
        .squeeze("variable", drop=True)
        .astype("float32")
    )
    return nwp - obs


@out_format(units="degrees")
def cos_wind_from_direction_ensctrl_error(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble control cosine wind direction
    """
    nwp = cos_wind_from_direction_ensctrl(data, stations, reftimes, leadtimes, **kwargs)
    obs = (
        data["obs"]
        .preproc.get("wind_from_direction")
        .pipe(lambda x: x * 2 * np.pi / 360)  # to radians
        .pipe(np.cos)
        .preproc.unstack_time(reftimes, leadtimes)
        .to_array(name="wind_speed")
        .squeeze("variable", drop=True)
        .astype("float32")
    )
    return nwp - obs


@out_format(units="degC")
def dew_point_depression_ens(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of dew point depression (T - T_d)
    """
    t = air_temperature_ens(data, stations, *args, **kwargs)
    t_d = dew_point_temperature_ens(data, stations, *args, **kwargs)
    return (t - t_d).astype("float32")


@out_format(units="degC")
def dew_point_depression_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of dew point depression (T - T_d)
    """
    tdep = dew_point_depression_ens(data, stations, reftimes, leadtimes, **kwargs)
    return tdep.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="degC")
def dew_point_depression_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run dew point depression (T - T_d)
    """
    tdep = dew_point_depression_ens(data, stations, reftimes, leadtimes, **kwargs)
    return (
        tdep.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@cache
def _dew_point_temperature_ens(
    data: Dict[str, xr.Dataset], stations, **kwargs
) -> xr.Dataset:
    """
    Ensemble of dew point temperature in °C
    """
    return (
        data["nwp"]
        .preproc.get("dew_point_temperature")
        .preproc.interp(stations, **kwargs)
        .pipe(lambda x: x - 273.15)  # convert to celsius
        .astype("float32")
    )


@out_format(units="degC")
def dew_point_temperature_ens(
    data: Dict[str, xr.Dataset], stations, reftimes=None, leadtimes=None, **kwargs
) -> xr.DataArray:
    """
    Ensemble of dew point temperature in °C
    """
    ens_data = _dew_point_temperature_ens(data, stations, **kwargs)
    ens_data = ens_data.preproc.align_time(reftimes, leadtimes)
    return ens_data


@out_format(units="degC")
def dew_point_temperature_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of dew point temperature in °C
    """
    td = dew_point_temperature_ens(data, stations, **kwargs)
    return td.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="degC")
def dew_point_temperature_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run dew point temperature in °C
    """
    td = dew_point_temperature_ens(data, stations, **kwargs)
    return (
        td.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="degC")
def equivalent_potential_temperature_ens(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Control run equivalent potential temperature in °C
    """
    t = air_temperature_ens(data, stations, *args, **kwargs)
    rh = relative_humidity_ens(data, stations, *args, **kwargs)
    p = surface_air_pressure_ens(data, stations, *args, **kwargs) / 100  # Pa to hPa
    return calc.equivalent_potential_temperature_from_t_rh_p(t, rh, p)


@out_format(units="degC")
def equivalent_potential_temperature_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run equivalent potential temperature in °C
    """
    theta_e = equivalent_potential_temperature_ens(data, stations, **kwargs)
    return (
        theta_e.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="degC")
def equivalent_potential_temperature_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run equivalent potential temperature in °C
    """
    theta_e = equivalent_potential_temperature_ens(data, stations, **kwargs)
    return (
        theta_e.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="W m-2")
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


@out_format(units="W m-2")
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


@out_format(units="W m-2")
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


@cache
def _eastward_wind_ens(data: Dict[str, xr.Dataset], stations, **kwargs) -> xr.Dataset:
    return (
        data["nwp"]
        .preproc.get("eastward_wind")
        .preproc.interp(stations, **kwargs)
        .astype("float32")
    )


@out_format(units="m s-1")
def eastward_wind_ens(
    data: Dict[str, xr.Dataset], stations, reftimes=None, leadtimes=None, **kwargs
) -> xr.DataArray:
    ens_data = _eastward_wind_ens(data, stations, **kwargs)
    ens_data = ens_data.preproc.align_time(reftimes, leadtimes)
    return ens_data


@out_format(units="m s-1")
def eastward_wind_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of eastward wind in m/s
    """
    u = eastward_wind_ens(data, stations, **kwargs)
    return u.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="m s-1")
def eastward_wind_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of eastward wind in m/s
    """
    u = eastward_wind_ens(data, stations, **kwargs)
    return (
        u.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format()  # don't use 'units=hours' to avoid encoding it as timedelta
def leadtime(data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs):
    """
    Extract leadtime in hours
    """
    if len(data["nwp"]) == 0:
        raise KeyError([])
    ds = data["nwp"]
    ds = ds.drop_vars(ds.data_vars)
    ds = ds.drop_dims(("x", "y", "realization"), errors="ignore")
    ds = ds.preproc.align_time(reftimes, leadtimes, return_source_leadtimes=True)
    ds = ds.reset_coords("source_leadtime").rename({"source_leadtime": "leadtime"})
    ds["leadtime"] = ds.leadtime // np.timedelta64(1, "h")
    return ds.astype("float32")


@out_format(units="m")
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


@cache
def _northward_wind_ens(data: Dict[str, xr.Dataset], stations, **kwargs) -> xr.Dataset:
    return (
        data["nwp"]
        .preproc.get("northward_wind")
        .preproc.interp(stations, **kwargs)
        .astype("float32")
    )


@out_format(units="m s-1")
def northward_wind_ens(
    data: Dict[str, xr.Dataset], stations, reftimes=None, leadtimes=None, **kwargs
) -> xr.DataArray:
    ens_data = _northward_wind_ens(data, stations, **kwargs)
    ens_data = ens_data.preproc.align_time(reftimes, leadtimes)
    return ens_data


@out_format(units="m s-1")
def northward_wind_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of northward wind in m/s
    """
    v = northward_wind_ens(data, stations, **kwargs)
    return v.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="m s-1")
def northward_wind_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of northward wind in m/s
    """
    v = northward_wind_ens(data, stations, **kwargs)
    return (
        v.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="degC")
def potential_temperature_ens(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of potential temperature in °C
    """
    t = air_temperature_ens(data, stations, *args, **kwargs)
    p = surface_air_pressure_ens(data, stations, *args, **kwargs) / 100  # Pa to hPa
    return calc.potential_temperature_from_t_and_p(t, p)


@out_format(units="degC")
def potential_temperature_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of potential temperature in °C
    """
    v = potential_temperature_ens(data, stations, **kwargs)
    return v.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="degC")
def potential_temperature_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of potential temperature in °C
    """
    v = potential_temperature_ens(data, stations, **kwargs)
    return (
        v.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="Pa")
def pressure_difference_BAS_LUG_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of pressure difference between Basel and Lugano in Pascal
    """
    p = surface_air_pressure_ens(data, stations, **kwargs).to_dataset()
    pdiff = p.sel(station=["BAS", "LUG"]).diff("station").squeeze("station", drop=True)
    return (
        pdiff.mean("realization")
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="Pa")
def pressure_difference_BAS_LUG_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of pressure difference between Basel and Lugano in Pascal
    """
    p = surface_air_pressure_ens(data, stations, **kwargs).to_dataset()
    pdiff = p.sel(station=["BAS", "LUG"]).diff("station").squeeze("station", drop=True)
    return (
        pdiff.isel(realization=0, drop=True)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="Pa")
def pressure_difference_GVE_GUT_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of pressure difference between Geneva and Güttingen in Pascal
    """
    p = surface_air_pressure_ens(data, stations, **kwargs).to_dataset()
    pdiff = p.sel(station=["GVE", "GUT"]).diff("station").squeeze("station", drop=True)
    return (
        pdiff.mean("realization")
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="Pa")
def pressure_difference_GVE_GUT_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of pressure difference between Geneva and Güttingen in Pascal
    """
    p = surface_air_pressure_ens(data, stations, **kwargs).to_dataset()
    pdiff = p.sel(station=["GVE", "GUT"]).diff("station").squeeze("station", drop=True)
    return (
        pdiff.isel(realization=0, drop=True)
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="%")
def relative_humidity_ens(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of relative humidity in %
    """
    e = water_vapor_pressure_ens(data, stations, *args, **kwargs)
    e_s = water_vapor_saturation_pressure(data, stations, *args, **kwargs)
    return (e / e_s * 100).astype("float32")


@out_format(units="%")
def relative_humidity_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of relative humidity in %
    """
    rh = relative_humidity_ens(data, stations, **kwargs)
    return rh.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="%")
def relative_humidity_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run relative humidity in %
    """
    rh = relative_humidity_ens(data, stations, **kwargs)
    return (
        rh.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format()
def sin_wind_from_direction_ens(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of sine wind direction
    """
    wdir = wind_from_direction_ens(data, stations, *args, **kwargs)
    return np.sin(wdir * 2 * np.pi / 360)


@out_format()
def sin_wind_from_direction_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of sine wind direction
    """
    wdir = sin_wind_from_direction_ens(data, stations, **kwargs)
    return wdir.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format()
def sin_wind_from_direction_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble control of sine wind direction
    """
    wdir = sin_wind_from_direction_ens(data, stations, **kwargs)
    return (
        wdir.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="degrees")
def sin_wind_from_direction_ensavg_error(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble average sine wind direction
    """
    nwp = sin_wind_from_direction_ensavg(data, stations, reftimes, leadtimes, **kwargs)
    obs = (
        data["obs"]
        .preproc.get("wind_from_direction")
        .pipe(lambda x: x * 2 * np.pi / 360)  # to radians
        .pipe(np.sin)
        .preproc.unstack_time(reftimes, leadtimes)
        .to_array(name="wind_speed")
        .squeeze("variable", drop=True)
        .astype("float32")
    )
    return nwp - obs


@out_format(units="degrees")
def sin_wind_from_direction_ensctrl_error(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble control sine wind direction
    """
    nwp = sin_wind_from_direction_ensctrl(data, stations, reftimes, leadtimes, **kwargs)
    obs = (
        data["obs"]
        .preproc.get("wind_from_direction")
        .pipe(lambda x: x * 2 * np.pi / 360)  # to radians
        .pipe(np.sin)
        .preproc.unstack_time(reftimes, leadtimes)
        .to_array(name="wind_speed")
        .squeeze("variable", drop=True)
        .astype("float32")
    )
    return nwp - obs


@cache
def _specific_humidity_ens(
    data: Dict[str, xr.Dataset], stations, **kwargs
) -> xr.Dataset:
    """
    Ensemble mean of specific humidity in g/kg
    """
    return (
        data["nwp"]
        .preproc.get("specific_humidity")
        .preproc.interp(stations, **kwargs)
        .astype("float32")
    )


@out_format(units="g kg-1")
def specific_humidity_ens(
    data: Dict[str, xr.Dataset], stations, reftimes=None, leadtimes=None, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of specific humidity in g/kg
    """
    ens_data = _specific_humidity_ens(data, stations, **kwargs)
    ens_data = ens_data.preproc.align_time(reftimes, leadtimes)
    return ens_data


@out_format(units="g kg-1")
def specific_humidity_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of specific humidity in g/kg
    """
    q = specific_humidity_ens(data, stations, **kwargs)
    return q.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="g kg-1")
def specific_humidity_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run specific humidity in g/kg
    """
    q = specific_humidity_ens(data, stations, **kwargs)
    return (
        q.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="s")
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


@cache
def _surface_air_pressure_ens(
    data: Dict[str, xr.Dataset], stations, **kwargs
) -> xr.Dataset:
    """
    Ensemble of surface pressure in Pascal
    """
    return (
        data["nwp"]
        .preproc.get("surface_air_pressure")
        .preproc.interp(stations, **kwargs)
        .astype("float32")
    )


@out_format(units="Pa")
def surface_air_pressure_ens(
    data: Dict[str, xr.Dataset], stations, reftimes=None, leadtimes=None, **kwargs
) -> xr.DataArray:
    """
    Ensemble of surface pressure in Pascal
    """
    ens_data = _surface_air_pressure_ens(data, stations, **kwargs)
    ens_data = ens_data.preproc.align_time(reftimes, leadtimes)
    return ens_data


@out_format(units="Pa")
def surface_air_pressure_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of surface pressure in Pascal
    """
    q = surface_air_pressure_ens(data, stations, **kwargs)
    return q.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="Pa")
def surface_air_pressure_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run surface pressure in Pascal
    """
    q = surface_air_pressure_ens(data, stations, **kwargs)
    return (
        q.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format()
def sx_500m_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Extract Sx with a 500m radius for the ensemble mean of wind direction.

    This uses azimuth sectors of 10 degrees, every 5 degrees.
    """
    sx = data["terrain"].preproc.get("SX_50M_RADIUS500")
    nsectors = sx.wind_from_direction.size
    degsector = int(360 / nsectors)

    # find correct index for every sample
    wdir = wind_from_direction_ensavg(data, stations, reftimes, leadtimes, **kwargs)
    wdir = wdir.load()
    is_valid = np.isfinite(wdir)
    wdir = wdir.astype("int16")
    ind = (wdir + int(degsector / 2)) // degsector
    ind = ind.astype("int8")
    ind = ind.where(ind != nsectors, 0)
    del wdir

    # compute Sx
    station_sub = stations.loc[ind.station]
    sx = sx.preproc.interp(station_sub)
    sx = sx.isel(wind_from_direction=ind.sel(station=sx.station))
    sx = sx.drop_vars("wind_from_direction")
    sx = sx.where(is_valid.sel(station=sx.station))

    return sx.astype("float32")


@out_format()
def sx_500m_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Extract Sx with a 500m radius for the ensemble control of wind direction.

    This uses azimuth sectors of 10 degrees, every 5 degrees.
    """
    sx = data["terrain"].preproc.get("SX_50M_RADIUS500")
    nsectors = sx.wind_from_direction.size
    degsector = int(360 / nsectors)

    # find correct index for every sample
    wdir = wind_from_direction_ensctrl(data, stations, reftimes, leadtimes, **kwargs)
    wdir = wdir.load()
    is_valid = np.isfinite(wdir)
    wdir = wdir.astype("int16")
    ind = (wdir + int(degsector / 2)) // degsector
    ind = ind.astype("int8")
    ind = ind.where(ind != nsectors, 0)
    del wdir

    # compute Sx
    station_sub = stations.loc[ind.station]
    sx = sx.preproc.interp(station_sub)
    sx = sx.isel(wind_from_direction=ind.sel(station=sx.station))
    sx = sx.drop_vars("wind_from_direction")
    sx = sx.where(is_valid.sel(station=sx.station))

    return sx.astype("float32")


@out_format(units="g kg-1")
def water_vapor_mixing_ratio_ens(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor mixing ratio in g/kg
    """
    e = water_vapor_pressure_ens(data, stations, *args, **kwargs)
    p = surface_air_pressure_ens(data, stations, *args, **kwargs) / 100  # Pa to hPa
    return calc.mixing_ratio_from_p_and_e(p, e)


@out_format(units="g kg-1")
def water_vapor_mixing_ratio_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor mixing ratio in g/kg
    """
    r = water_vapor_mixing_ratio_ens(data, stations, **kwargs)
    return r.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="g kg-1")
def water_vapor_mixing_ratio_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run water vapor mixing ratio in g/kg
    """
    r = water_vapor_mixing_ratio_ens(data, stations, **kwargs)
    return (
        r.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="hPa")
def water_vapor_pressure_ens(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Ensemble of water vapor partial pressure in hPa
    """
    t_d = dew_point_temperature_ens(data, stations, *args, **kwargs)
    t = air_temperature_ens(data, stations, *args, **kwargs)
    return calc.water_vapor_pressure_from_t_and_td(t, t_d)


@out_format(units="hPa")
def water_vapor_pressure_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor partial pressure
    """
    e = water_vapor_pressure_ens(data, stations, **kwargs)
    return e.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="hPa")
def water_vapor_pressure_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run water vapor partial pressure
    """
    e = water_vapor_pressure_ens(data, stations, **kwargs)
    return (
        e.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="hPa")
def water_vapor_saturation_pressure(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor partial pressure at saturation in hPa
    """
    t = air_temperature_ens(data, stations, *args, **kwargs)
    return calc.water_vapor_saturation_pressure_from_t(t)


@out_format(units="hPa")
def water_vapor_saturation_pressure_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of water vapor partial pressure at saturation in hPa
    """
    e_s = water_vapor_saturation_pressure(data, stations, **kwargs)
    return e_s.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="hPa")
def water_vapor_saturation_pressure_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Control run water vapor partial pressure at saturation
    """
    e_s = water_vapor_saturation_pressure(data, stations, **kwargs)
    return (
        e_s.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="degrees")
def wind_from_direction_ens(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble control of wind direction
    """
    u = eastward_wind_ens(data, stations, *args, **kwargs)
    v = northward_wind_ens(data, stations, *args, **kwargs)
    out = (270 - 180 / np.pi * np.arctan2(v, u)) % 360
    return out.astype("float32")


@out_format(units="degrees")
def wind_from_direction_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of wind direction
    """
    d = wind_from_direction_ens(data, stations, **kwargs)
    return d.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="degrees")
def wind_from_direction_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble control of wind direction
    """
    d = wind_from_direction_ens(data, stations, **kwargs)
    return (
        d.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="rank")
def wind_from_direction_rank(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate rank of ensemble of wind direction
    """
    d = wind_from_direction_ens(data, stations, **kwargs)
    return (
        d.to_dataset()
        .preproc.rankdata(dim="realization", circular=True)
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="m s-1")
def wind_speed_ens(data: Dict[str, xr.Dataset], stations, *args, **kwargs):
    u = eastward_wind_ens(data, stations, *args, **kwargs)
    v = northward_wind_ens(data, stations, *args, **kwargs)
    return np.sqrt(u**2 + v**2).astype("float32")


@out_format(units="m s-1")
def wind_speed_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of wind speed
    """
    uv = wind_speed_ens(data, stations, **kwargs)
    return uv.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="m s-1")
def wind_speed_ensavg_error(
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
        .astype("float32")
    )
    return nwp - obs


@out_format(units="m s-1")
def wind_speed_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of wind speed
    """
    uv = wind_speed_ens(data, stations, **kwargs)
    return (
        uv.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="m s-1")
def wind_speed_ensctrl_3hmean(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of 3h mean wind speed
    """
    uv = wind_speed_ens(data, stations, **kwargs)
    return (
        uv.isel(realization=0, drop=True)
        .rolling(t=3, center=True, min_periods=1)
        .mean()
        .astype("float32")
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="m s-1")
def wind_speed_ensctrl_5hmean(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of 5h mean wind speed
    """
    uv = wind_speed_ens(data, stations, **kwargs)
    return (
        uv.isel(realization=0, drop=True)
        .rolling(t=5, center=True, min_periods=1)
        .mean()
        .astype("float32")
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="m s-1")
def wind_speed_ensctrl_error(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Forecast error of the ensemble control wind speed
    """
    nwp = wind_speed_ensctrl(data, stations, reftimes, leadtimes, **kwargs)
    obs = data["obs"][["wind_speed"]]
    obs = (
        obs.preproc.unstack_time(reftimes, leadtimes)
        .to_array(name="wind_speed")
        .squeeze("variable", drop=True)
        .astype("float32")
    )
    return nwp - obs


@out_format(units="m s-1")
def wind_speed_ensstd(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble std of wind speed
    """
    uv = wind_speed_ens(data, stations, **kwargs)
    return uv.std("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format()
def wind_speed_enscov(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble coefficient of variation of wind speed
    """
    mean = wind_speed_ensavg(data, stations, *args, **kwargs)
    std = wind_speed_ensstd(data, stations, *args, **kwargs)
    return (std + 0.1) / (mean + 0.1)


@out_format(units="rank")
def wind_speed_rank(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate rank of ensemble of wind speed
    """
    d = wind_speed_ens(data, stations, **kwargs)
    return (
        d.to_dataset()
        .preproc.rankdata(dim="realization")
        .preproc.align_time(reftimes, leadtimes)
    )


@cache
def _wind_speed_of_gust_ens(
    data: Dict[str, xr.Dataset], stations, **kwargs
) -> xr.Dataset:
    """
    Ensemble of wind speed gust
    """
    return (
        data["nwp"]
        .preproc.get("wind_speed_of_gust")
        .preproc.interp(stations, **kwargs)
        .astype("float32")
    )


@out_format(units="m s-1")
def wind_speed_of_gust_ens(
    data: Dict[str, xr.Dataset], stations, reftimes=None, leadtimes=None, **kwargs
) -> xr.DataArray:
    """
    Ensemble of wind speed gust
    """
    ens_data = _wind_speed_of_gust_ens(data, stations, **kwargs)
    ens_data = ens_data.preproc.align_time(reftimes, leadtimes)
    return ens_data


@out_format(units="m s-1")
def wind_speed_of_gust_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble mean of wind speed gust
    """
    ug = wind_speed_of_gust_ens(data, stations, **kwargs)
    return ug.mean("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format(units="m s-1")
def wind_speed_of_gust_ensavg_error(
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
        .astype("float32")
    )
    return nwp - obs


@out_format(units="m s-1")
def wind_speed_of_gust_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of wind speed of gust
    """
    ug = wind_speed_of_gust_ens(data, stations, **kwargs)
    return (
        ug.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format(units="m s-1")
def wind_speed_of_gust_ensctrl_3hmean(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of 3h mean hourly wind speed of gust
    """
    ug = wind_speed_of_gust_ens(data, stations, **kwargs)
    return (
        ug.isel(realization=0, drop=True)
        .rolling(t=3, center=True, min_periods=1)
        .mean()
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m s-1")
def wind_speed_of_gust_ensctrl_5hmean(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble control of 5h mean hourly wind speed of gust
    """
    ug = wind_speed_of_gust_ens(data, stations, **kwargs)
    return (
        ug.isel(realization=0, drop=True)
        .rolling(t=3, center=True, min_periods=1)
        .mean()
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m s-1")
def wind_speed_of_gust_ensctrl_error(
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
        .astype("float32")
    )
    return nwp - obs


@out_format(units="m s-1")
def wind_speed_of_gust_ensstd(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble std of wind speed gust
    """
    ug = wind_speed_of_gust_ens(data, stations, **kwargs)
    return ug.std("realization").to_dataset().preproc.align_time(reftimes, leadtimes)


@out_format()
def wind_speed_of_gust_enscov(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble coefficient of variation of wind gust
    """
    mean = wind_speed_of_gust_ensavg(data, stations, *args, **kwargs)
    std = wind_speed_of_gust_ensstd(data, stations, *args, **kwargs)
    return (std + 0.1) / (mean + 0.1)


@out_format(units="rank")
def wind_speed_of_gust_rank(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate rank of ensemble of wind speed of gust
    """
    d = wind_speed_of_gust_ens(data, stations, **kwargs)
    return (
        d.to_dataset()
        .preproc.rankdata(dim="realization")
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format()
def wind_gust_factor_ens(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble of wind gust factor
    """
    speed = wind_speed_ens(data, stations, *args, **kwargs)
    gust = wind_speed_of_gust_ens(data, stations, *args, **kwargs)
    return (gust + 1.0) / (speed + 1.0)


@out_format()
def wind_gust_factor_ensavg(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble mean of wind gust factor
    """
    gust_factor = wind_gust_factor_ens(data, stations, **kwargs)
    return (
        gust_factor.mean("realization")
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )


@out_format()
def wind_gust_factor_ensctrl(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Calculate ensemble control of wind gust factor
    """
    gust_factor = wind_gust_factor_ens(data, stations, **kwargs)
    return (
        gust_factor.isel(realization=0, drop=True)
        .to_dataset()
        .preproc.align_time(reftimes, leadtimes)
    )
