import logging
from typing import Dict

import xarray as xr
import numpy as np

from mlpp_features.decorators import asarray

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@asarray
def cos_dayofyear(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, ds, **kwargs
) -> xr.Dataset:
    """
    Compute the cosine of day-of-year
    """
    if data["nwp"] is not None and len(data["nwp"]) == 0:
        raise KeyError([])
    ds = xr.Dataset(
        None,
        coords={
            "forecast_reference_time": reftimes,
            "t": leadtimes,
        },
    )
    ds = ds.assign_coords(
        time=ds.forecast_reference_time + ds.t.astype("timedelta64[h]")
    )
    ds["cos_dayofyear"] = (
        (ds["time.dayofyear"] + ds["time.hour"] / 24) * 2 * np.pi / 366
    )
    return ds.pipe(np.cos).astype("float32")


@asarray
def cos_hourofday(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, ds, **kwargs
) -> xr.Dataset:
    """
    Compute the cosine of hour-of-day
    """
    if data["nwp"] is not None and len(data["nwp"]) == 0:
        raise KeyError([])
    ds = xr.Dataset(
        None,
        coords={
            "forecast_reference_time": reftimes,
            "t": leadtimes,
        },
    )
    ds = ds.assign_coords(
        time=ds.forecast_reference_time + ds.t.astype("timedelta64[h]")
    )
    ds["cos_hourofday"] = ds["time.hour"] * 2 * np.pi / 24
    return ds.pipe(np.cos).astype("float32")


@asarray
def sin_dayofyear(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, ds, **kwargs
) -> xr.Dataset:
    """
    Compute the sine of day-of-year
    """
    if data["nwp"] is not None and len(data["nwp"]) == 0:
        raise KeyError([])
    ds = _make_time_dataset(reftimes, leadtimes)
    ds["sin_dayofyear"] = (
        (ds["time.dayofyear"] + ds["time.hour"] / 24) * 2 * np.pi / 366
    )
    return ds.pipe(np.sin).astype("float32")


@asarray
def sin_hourofday(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, ds, **kwargs
) -> xr.Dataset:
    """
    Compute the sine of hour-of-day
    """
    if data["nwp"] is not None and len(data["nwp"]) == 0:
        raise KeyError([])
    ds = _make_time_dataset(reftimes, leadtimes)
    ds["sin_hourofday"] = ds["time.hour"] * 2 * np.pi / 24
    return ds.pipe(np.cos).astype("float32")


def _make_time_dataset(reftimes, leadtimes):
    ds = xr.Dataset(
        None,
        coords={
            "forecast_reference_time": reftimes,
            "t": leadtimes,
        },
    )
    return ds.assign_coords(
        time=ds.forecast_reference_time + ds.t.astype("timedelta64[h]")
    )
