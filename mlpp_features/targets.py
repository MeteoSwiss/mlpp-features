import logging
from typing import Dict

import xarray as xr
import numpy as np

from mlpp_features.decorators import asarray

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@asarray
def wind_speed(data: xr.Dataset) -> xr.DataArray:
    return data["obs"]["wind_speed"].astype("float32")
