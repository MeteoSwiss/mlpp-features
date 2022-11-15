import numpy as np
import xarray as xr

EPSILON = 0.622
A_P = 17.368
A_N = 17.856
B_P = 238.83
B_N = 245.52
C_P = 6.107
C_N = 6.108

R = 287.053  # gas constant for dry air
Cp = 1004.0  # specific heat of dry air at costant pressure
P0 = 1000.0  # standard reference pressure in hPa


def dew_point_from_t_and_rh(t: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
    """
    Compute dew point temperature in °C from temperature (°C)
    and relative humidity (%).
    """
    e = water_vapor_pressure_from_t_and_rh(t, rh)
    return xr.where(
        t >= 0.0,
        (B_P * np.log(e / C_P)) / (A_P - np.log(e / C_P)),
        (B_N * np.log(e / C_N)) / (A_N - np.log(e / C_N)),
    )


def equivalent_potential_temperature_from_t_rh_p(
    t: xr.DataArray, rh: xr.DataArray, p: xr.DataArray
) -> xr.DataArray:
    l_v = (
        -3.07 * t + 2477.0
    )  # latent heat of vaporization, linear approximation of Lv(T)
    r = mixing_ratio_from_t_rh_p(t, rh, p)
    t_e = t + l_v / Cp * r
    return potential_temperature_from_t_and_p(t_e, p)


def mixing_ratio_from_t_rh_p(
    t: xr.DataArray, rh: xr.DataArray, p: xr.DataArray
) -> xr.DataArray:
    """
    Compute water vapor mixing ratio in g kg-1 from temperature (°C),
    relative humidity (%) and pressure (hPa).
    """
    e = water_vapor_pressure_from_t_and_rh(t, rh)
    return mixing_ratio_from_p_and_e(p, e)


def mixing_ratio_from_p_and_e(p: xr.DataArray, e: xr.DataArray) -> xr.DataArray:
    """
    Compute water vapor mixing ratio in g kg-1 from pressure (hPa)
    and water vapor pressure (hPa).
    """
    return 1000 * (EPSILON * e) / (p - e)


def potential_temperature_from_t_and_p(
    t: xr.DataArray, p: xr.DataArray
) -> xr.DataArray:
    """
    Compute potential temperature in °C from temperature (°C) and pressure (hPa).
    """
    t_kelvin = t + 273.15
    return t_kelvin * (P0 / p) ** (R / Cp) - 273.15


def water_vapor_saturation_pressure_from_t(t: xr.DataArray) -> xr.DataArray:
    """
    Compute water vapor pressure at saturation in hPa from temperature (°C).
    """
    return xr.where(
        t >= 0.0,
        C_P * np.exp((A_P * t) / (t + B_P)),
        C_N * np.exp((A_N * t) / (t + B_N)),
    )


def water_vapor_pressure_from_t_and_td(
    t: xr.DataArray, t_d: xr.DataArray
) -> xr.DataArray:
    """
    Compute water vapor pressure at saturation in hPa from temperature (°C) and dew point temperature (°C).
    """
    return xr.where(
        t >= 0.0,
        C_P * np.exp((A_P * t_d) / (t_d + B_P)),
        C_N * np.exp((A_N * t_d) / (t_d + B_N)),
    )


def water_vapor_pressure_from_t_and_rh(
    t: xr.DataArray, rh: xr.DataArray
) -> xr.DataArray:
    """
    Compute water vapor pressure at saturation in hPa from temperature (°C).
    """
    e_s = water_vapor_saturation_pressure_from_t(t)
    return e_s * rh / 100
