import logging
from typing import Dict

import pandas as pd
import xarray as xr

from mlpp_features.decorators import out_format

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


# @out_format()
# def cloud_area_fraction_rollingmean_1h_10d(
#     data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
# ) -> xr.DataArray:
#     """
#     Climatology of cloud cover: the values were averaged at +-1h and +-10 days to account for seasonal variability 
#     """
#     return (
#         data["climatology"]
#         .mlpp.get("cloud_area_fraction")
#         .mlpp.unstack_time(reftimes, leadtimes)
#         .astype("float32")
#     )


@out_format()
def cloud_area_fraction_rollingmean_1h_10d(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Climatology of cloud cover: the values were averaged at +-1h and +-10 days to account for seasonal variability 
    """
    rolling_mean_ds = (
        data["obs"]
        .mlpp.get("cloud_area_fraction")
        .mlpp.rolling_mean_days_and_hours(hours=1, days=10)
    )
    clim_ds = xr.Dataset(
        None,
        coords={
            "forecast_reference_time": pd.to_datetime(reftimes).astype("datetime64[ns]"),
            "lead_time": leadtimes.astype("timedelta64[ns]"),
        },
    )

    clim_ds = clim_ds.assign_coords(time=clim_ds.forecast_reference_time + clim_ds.lead_time)
    days = clim_ds.time.dt.dayofyear
    hours = clim_ds.time.dt.hour

    clim_ds = (
        rolling_mean_ds.sel(dayofyear=days, hourofday=hours)
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )

    return clim_ds