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
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.Dataset:
    """
    Compute the cosine of day-of-year
    """
    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["time"]
    except KeyError:
        raise KeyError(["time"])

    ds = data["nwp"][["time"]]
    ds["cos_dayofyear"] = (
        (ds["time.dayofyear"] + ds["time.hour"] / 24) * 2 * np.pi / 366
    )
    return ds.pipe(np.cos).preproc.align_time(reftimes, leadtimes).astype("float32")


@asarray
def cos_hourofday(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.Dataset:
    """
    Compute the cosine of hour-of-day
    """
    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["time"]
    except KeyError:
        raise KeyError(["time"])

    ds = data["nwp"][["time"]]
    ds["cos_hourofday"] = ds["time.hour"] * 2 * np.pi / 24
    return ds.pipe(np.cos).preproc.align_time(reftimes, leadtimes).astype("float32")


@asarray
def sin_dayofyear(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.Dataset:
    """
    Compute the sine of day-of-year
    """
    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["time"]
    except KeyError:
        raise KeyError(["time"])

    ds = data["nwp"][["time"]]
    ds["sin_dayofyear"] = (
        (ds["time.dayofyear"] + ds["time.hour"] / 24) * 2 * np.pi / 366
    )
    return ds.pipe(np.sin).preproc.align_time(reftimes, leadtimes).astype("float32")


@asarray
def sin_hourofday(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.Dataset:
    """
    Compute the sine of hour-of-day
    """
    # try/except block necessary to expose all the required input data
    try:
        data["nwp"]["time"]
    except KeyError:
        raise KeyError(["time"])

    ds = data["nwp"][["time"]]
    ds["sin_hourofday"] = ds["time.hour"] * 2 * np.pi / 24
    return ds.pipe(np.cos).preproc.align_time(reftimes, leadtimes).astype("float32")
