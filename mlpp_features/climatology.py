import logging
from typing import Dict

import xarray as xr
import numpy as np

from mlpp_features.decorators import cache, inputs, out_format
from mlpp_features import calc

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@out_format()
def cloud_area_fraction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Climatology of cloud cover: the values were averaged at +-1h and +-10 days to account for seasonal variability 
    """
    return (
        data["climatology"]
        .mlpp.get("cloud_area_fraction")
        .mlpp.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )
