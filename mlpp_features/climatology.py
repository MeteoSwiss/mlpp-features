import logging
from typing import Dict

import xarray as xr

from mlpp_features.decorators import out_format

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@out_format()
def cloud_area_fraction_rollingmean_1h_10d(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Climatology of cloud cover: the values were averaged at +-1h and +-10 days to account for seasonal variability 
    """
    return (
        data["climatology"]
        .mlpp.get("cloud_area_fraction")
        .mlpp.interp(stations, **kwargs)
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )
