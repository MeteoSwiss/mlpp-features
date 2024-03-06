import logging
from datetime import datetime
from typing import Dict

import xarray as xr
import numpy as np
import pandas as pd

from mlpp_features.decorators import inputs, out_format

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


def _make_time_dataset(reftimes, leadtimes):
    ds = xr.Dataset(
        None,
        coords={
            "forecast_reference_time": reftimes.astype("datetime64[ns]"),
            "lead_time": leadtimes.astype("timedelta64[ns]"),
        },
    )
    return ds.assign_coords(time=ds.forecast_reference_time + ds.lead_time)


@inputs()
@out_format()
def cos_dayofyear(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.Dataset:
    """
    Compute the cosine of day-of-year
    """
    ds = xr.Dataset(
        None,
        coords={
            "forecast_reference_time": reftimes.astype("datetime64[ns]"),
            "lead_time": leadtimes.astype("timedelta64[ns]"),
        },
    )
    ds = ds.assign_coords(time=ds.forecast_reference_time + ds.lead_time)
    ds["cos_dayofyear"] = (
        (ds["time.dayofyear"] + ds["time.hour"] / 24) * 2 * np.pi / 366
    )
    return ds.pipe(np.cos).astype("float32")


@inputs()
@out_format()
def cos_hourofday(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.Dataset:
    """
    Compute the cosine of hour-of-day
    """
    ds = xr.Dataset(
        None,
        coords={
            "forecast_reference_time": reftimes.astype("datetime64[ns]"),
            "lead_time": leadtimes.astype("timedelta64[ns]"),
        },
    )
    ds = ds.assign_coords(time=ds.forecast_reference_time + ds.lead_time)
    ds["cos_hourofday"] = ds["time.hour"] * 2 * np.pi / 24
    return ds.pipe(np.cos).astype("float32")


@inputs()
@out_format()
def sin_dayofyear(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.Dataset:
    """
    Compute the sine of day-of-year
    """
    ds = _make_time_dataset(reftimes, leadtimes)
    ds["sin_dayofyear"] = (
        (ds["time.dayofyear"] + ds["time.hour"] / 24) * 2 * np.pi / 366
    )
    return ds.pipe(np.sin).astype("float32")


@inputs()
@out_format()
def sin_hourofday(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.Dataset:
    """
    Compute the sine of hour-of-day
    """
    ds = _make_time_dataset(reftimes, leadtimes)
    ds["sin_hourofday"] = ds["time.hour"] * 2 * np.pi / 24
    return ds.pipe(np.sin).astype("float32")


@inputs()
@out_format()
def weight_sample_age(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.Dataset:
    """
    Compute the inverse of the sample age in years
    """
    ds = _make_time_dataset(reftimes, leadtimes)
    this_year = datetime.today().year * 365 + datetime.today().timetuple().tm_yday
    ds["weight_sample_age"] = (
        2.0 / (1 + this_year - ds["time.year"] * 365 - ds["time.dayofyear"]) ** 0.5
    )
    return ds.astype("float32")


@inputs()
@out_format()
def weight_leadtime(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.Dataset:
    """
    Weight the lead time.
    """
    weight_leadtime = 1.5 / (1 + leadtimes / pd.Timedelta(hours=24))
    ds = xr.Dataset(
        {"weight_leadtime": ("lead_time", weight_leadtime)},
        coords={"lead_time": leadtimes.astype("timedelta64[ns]")},
    )
    return ds.astype("float32")
