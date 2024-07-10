import logging
from typing import Dict

import numpy as np
import xarray as xr

from mlpp_features.decorators import inputs, out_format


_LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@out_format()
def cloud_area_fraction(
    data: Dict[str, xr.Dataset], stations, reftimes, leadtimes, **kwargs
) -> xr.DataArray:
    """
    Ensemble of total cloud cover in %
    """
    return (
        data["sat"]
        .preproc.get("cloud_area_fraction")
        .preproc.unstack_time(reftimes, leadtimes)
        .astype("float32")
    )