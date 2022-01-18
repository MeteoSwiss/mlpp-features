""""""
import logging
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

import mlpp_features.point_selection as ps


LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)


@xr.register_dataset_accessor("preproc")
@dataclass
class PreprocDatasetAccessor:
    """
    Access methods for Datasets with preprocessing methods.
    """

    ds: xr.Dataset
    selector: ps.PointSelector = field(init=False, repr=True)

    def __post_init__(self):
        if "lat" in self.ds:
            self.selector = ps.EuclideanNearestIrregular(self.ds)
        else:
            self.selector = ps.EuclideanNearestRegular(self.ds)

    def get(self, var):
        """
        Get one or more variables from a Dataset.
        """
        try:
            out = self.ds[var]
        except KeyError:
            raise KeyError(var)
        try:
            return out.to_dataset(name=var)
        except AttributeError:
            return out
        except ValueError:
            return out.coords.to_dataset().reset_coords(var)

    def interp(self, *points, **kwargs):
        """
        Interpolate all variables in the dataset onto a set of target points.
        """
        point_names = points[0]
        point_coords = points[1:]
        index, mask = self.selector.query(point_coords, **kwargs)
        ds_out = self.ds.stack(point=("y", "x")).isel(point=index).reset_index("point")
        point_names = [p for p, m in zip(point_names, mask) if m]
        ds_out = ds_out.assign_coords({"point": point_names})
        return ds_out

    def norm(self):
        """
        Compute the Euclidean norm of all variables in the input dataset.
        """
        vars = list(self.ds.keys())
        ds = xr.Dataset()
        ds["norm"] = xr.full_like(self.ds[vars[0]], fill_value=0)
        for var in vars:
            ds["norm"] += self.ds[var] ** 2
        return np.sqrt(ds[["norm"]])
