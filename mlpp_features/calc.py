import numpy as np 
import xarray as xr 

EPSILON = 0.622
A_P = 17.368
A_N = 17.856
B_P = 238.83
B_N = 245.52
C_P = 6.107
C_N = 6.108

def water_vapor_saturation_pressure_from_t(t: xr.DataArray) -> xr.DataArray:
    """
    Compute water vapor pressure at saturation in hPa from temperature.
    """
    return xr.where(
        t >= 0.,
        C_P * np.exp((A_P * t) / (t + B_P)),
        C_N * np.exp((A_N * t) / (t + B_N)),
    )


def water_vapor_pressure_from_t_and_rh(t: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
    """
    Compute water vapor pressure at saturation in hPa from temperature.
    """
    e_s = water_vapor_saturation_pressure_from_t(t)
    return e_s * rh / 100


def dew_point_from_t_and_rh(t: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
    """
    Compute dew point temperature in °C from temperature (°C) 
    and relative humidity (%).
    """
    e = water_vapor_pressure_from_t_and_rh(t, rh)
    return xr.where(
        t >= 0.,
        (B_P * np.log(e / C_P)) / (A_P - np.log(e / C_P)),
        (B_N * np.log(e / C_N)) / (A_N - np.log(e / C_N))
    )


def mixing_ratio_from_t_rh_p(t: xr.DataArray, rh: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    """
    Compute water vapor mixing ratio in g kg-1 from temperature (°C),
    relative humidity (%) and pressure (hPa).
    """
    e = water_vapor_pressure_from_t_and_rh(t, rh)
    return 1000 * (EPSILON * e) / (p - e)