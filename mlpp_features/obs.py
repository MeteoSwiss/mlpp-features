import logging
from typing import Dict

import numpy as np
import xarray as xr

from mlpp_features.decorators import inputs, out_format
from mlpp_features import calc

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@out_format(units="degC")
def air_temperature(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed air temperature in °C
    """
    return (
        data["obs"]
        .mlpp.get("air_temperature")
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format()
def cloud_area_fraction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble of total cloud cover (fraction)
    """
    return (
        data["obs"]
        .mlpp.get("cloud_area_fraction")
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@inputs("obs:air_temperature", "obs:dew_point_temperature")
@out_format(units="degC")
def dew_point_depression(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed dew point depression (T - T_d)
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    t_d = dew_point_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return (t - t_d).astype("float32")


@inputs("obs:air_temperature", "obs:relative_humidity")
@out_format(units="degC")
def dew_point_temperature(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed dew point temperature in °C
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    rh = relative_humidity(data, stations, reftimes, leadtimes, **kwargs)
    t_d = calc.dew_point_from_t_and_rh(t, rh)
    return t_d


@out_format(units="hPa")
def surface_air_pressure(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed surface pressure in hPa
    """
    return (
        data["obs"]
        .mlpp.get("surface_air_pressure")
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="%")
def relative_humidity(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed relative humidity in %
    """
    return (
        data["obs"]
        .mlpp.get("relative_humidity")
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@inputs("obs:air_temperature", "obs:relative_humidity", "obs:surface_air_pressure")
@out_format(units="g kg-1")
def water_vapor_mixing_ratio(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed water vapor mixing ratio in g/kg
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    rh = relative_humidity(data, stations, reftimes, leadtimes, **kwargs)
    p = surface_air_pressure(data, stations, reftimes, leadtimes, **kwargs)
    r = calc.mixing_ratio_from_t_rh_p(t, rh, p)
    return r


@out_format()
def cos_wind_from_direction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Cosine of observed wind directions
    """
    return (
        data["obs"]
        .mlpp.get("wind_from_direction")
        .pipe(lambda x: x * 2 * np.pi / 360)  # to radians
        .pipe(np.cos)
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format()
def cos_wind_from_direction_3hmean(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    cosine of observed wind directions averaged over 3 hours
    """
    return (
        data["obs"]
        .mlpp.get("wind_from_direction")
        .pipe(lambda x: x * 2 * np.pi / 360)  # to radians
        .pipe(np.cos)
        .rolling(time=3, center=True, min_periods=1)
        .mean()
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format()
def sin_wind_from_direction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Sine of observed wind directions
    """
    return (
        data["obs"]
        .mlpp.get("wind_from_direction")
        .pipe(lambda x: x * 2 * np.pi / 360)  # to radians
        .pipe(np.sin)
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format()
def sin_wind_from_direction_3hmean(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    sine of observed wind directions averaged over 3 hours
    """
    return (
        data["obs"]
        .mlpp.get("wind_from_direction")
        .pipe(lambda x: x * 2 * np.pi / 360)  # to radians
        .pipe(np.sin)
        .rolling(time=3, center=True, min_periods=1)
        .mean()
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@inputs("obs:air_temperature", "obs:relative_humidity")
@out_format(units="hPa")
def water_vapor_pressure(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
):
    """
    Water vapor pressure in hPa
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    rh = relative_humidity(data, stations, reftimes, leadtimes, **kwargs)
    return calc.water_vapor_pressure_from_t_and_rh(t, rh)


@out_format(units="hPa")
def water_vapor_saturation_pressure(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
):
    """
    Water vapor pressure at saturation in hPa
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    return calc.water_vapor_saturation_pressure_from_t(t)


@out_format()
def wind_from_direction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed wind directions
    """
    return (
        data["obs"]
        .mlpp.get("wind_from_direction")
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m s-1")
def wind_speed(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed wind speed in m/s
    """
    return (
        data["obs"]
        .mlpp.get("wind_speed")
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m s-1")
def wind_speed_3hmax(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed 3-hourly max wind speed in m/s
    """
    return (
        data["obs"]
        .mlpp.get("wind_speed")
        .rolling(time=3, center=True, min_periods=1)
        .max()
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m s-1")
def wind_speed_3hmean(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed 3-hourly mean wind speed in m/s
    """
    return (
        data["obs"]
        .mlpp.get("wind_speed")
        .rolling(time=3, center=True, min_periods=1)
        .mean()
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m s-1")
def wind_speed_of_gust(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed wind gust in m/s
    """
    return (
        data["obs"]
        .mlpp.get("wind_speed_of_gust")
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m s-1")
def wind_speed_of_gust_3hmax(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed 3-hourly max wind gust in m/s
    """
    return (
        data["obs"]
        .mlpp.get("wind_speed_of_gust")
        .rolling(time=3, center=True, min_periods=1)
        .max()
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )


@inputs("obs:air_temperature", "obs:surface_air_pressure")
@out_format(units="degC")
def potential_temperature(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed potential temperature in °C
    """
    t = air_temperature(data, stations, reftimes, leadtimes, **kwargs)
    p = surface_air_pressure(data, stations, reftimes, leadtimes, **kwargs)
    return calc.potential_temperature_from_t_and_p(t, p)


@out_format(units="m s-1")
def nearest_wind_speed(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed wind speed at the nearest (euclidean distance) station
    """
    return (
        data["obs"]
        .mlpp.get("wind_speed")
        .mlpp.euclidean_nearest_k(stations, k=5)
        .mlpp.select_rank(rank=1)
        .mlpp.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m")
def distance_to_nearest_wind_speed(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Distance (euclidean, in meters) from the nearest wind speed measurement.
    """
    return (
        data["obs"]
        .mlpp.get("wind_speed")
        .mlpp.euclidean_nearest_k(stations, k=5)
        .mlpp.select_rank(rank=1)
        .drop_vars("wind_speed")
        .reset_coords("neighbor_1_distance")
        .mlpp.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m s-1")
def nearest_wind_speed_of_gust(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Observed wind gust at the nearest (euclidean distance) station
    """
    return (
        data["obs"]
        .mlpp.get("wind_speed_of_gust")
        .mlpp.euclidean_nearest_k(stations, k=5)
        .mlpp.select_rank(rank=1)
        .mlpp.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@out_format(units="m")
def distance_to_nearest_wind_speed_of_gust(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Distance (euclidean, in meters) from the nearest wind gust measurement.
    """
    return (
        data["obs"]
        .mlpp.get("wind_speed_of_gust")
        .mlpp.euclidean_nearest_k(stations, k=5)
        .mlpp.select_rank(rank=1)
        .drop_vars("wind_speed_of_gust")
        .reset_coords("neighbor_1_distance")
        .mlpp.persist_observations(reftimes, leadtimes)
        .astype("float32")
    )


@inputs()
@out_format()
def weight_owner_id(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Weight the station owner.
    """
    owner_id = stations.owner_id.to_xarray()
    owner_weight = xr.full_like(owner_id, 1.0)
    owner_weight = owner_weight.where(owner_id > 1, 2)
    ds = xr.Dataset(
        {"weight_owner_id": ("station", owner_weight.data)},
        coords={"station": stations.index},
    )
    return ds.astype("float32")


@inputs()
@out_format()
def measurement_height(
    data: Dict[str, xr.Dataset], stations, *args, **kwargs
) -> xr.Dataset:
    """
    Weight the station owner.
    """
    pole_height = stations.pole_height.to_xarray()
    fillvalue_pole_height = pole_height.median()
    LOGGER.debug(f"Fill value pole height: {fillvalue_pole_height:.1f}")
    pole_height = pole_height.fillna(fillvalue_pole_height)
    roof_height = stations.roof_height.to_xarray()
    fillvalue_roof_height = roof_height.median()
    LOGGER.debug(f"Fill value roof height: {fillvalue_roof_height:.1f}")
    roof_height = roof_height.fillna(fillvalue_roof_height)
    ds = xr.Dataset(
        {"measurement_height": ("station", pole_height.data + roof_height.data)},
        coords={"station": stations.index},
    )
    return ds.astype("float32")
