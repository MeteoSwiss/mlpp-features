""""""
import logging
from dataclasses import dataclass, field
from typing import List, Union

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

    def get(self, var: Union[str, List[str]]) -> xr.Dataset:
        """
        Get one or more variables from a Dataset.
        """
        if isinstance(var, str):
            var = [var]
        try:
            return self.ds[var]
        except KeyError:
            raise KeyError(var)

    def interp(self, points, **kwargs):
        """
        Interpolate all variables in the dataset onto a set of target points.
        """

        if "station_id" in self.ds:
            selector = ps.EuclideanNearestSparse(self.ds)
        elif "latitude" in self.ds:
            selector = ps.EuclideanNearestIrregular(self.ds)
        else:
            selector = ps.EuclideanNearestRegular(self.ds)

        point_names = points[0]
        point_coords = points[1:]
        index, mask = selector.query(point_coords, **kwargs)
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

    def difference(self, var1: str, var2: str) -> xr.Dataset:
        """
        Compute the difference between two variables in the input dataset
        """
        return (self.ds[var1] - self.ds[var2]).to_dataset(name="difference")

    def apply_func(self, f):
        """
        Apply a function to the input dataset.
        """
        return f(self.ds)

    def asarray(self, name):
        """
        Convert the input dataset to a dataarray with a new name.
        """
        return self.ds.to_array(name=name)[0].drop_vars("variable")
