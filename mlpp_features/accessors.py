""""""
import logging
from dataclasses import dataclass, field
from datetime import datetime
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
    selector: ps.PointSelector = field(init=False, repr=False)

    def __post_init__(self):

        if "station_id" in self.ds:
            self.selector = ps.EuclideanNearestSparse(self.ds)

    def get(self, var: Union[str, List[str]]) -> xr.Dataset:
        """Get one or more variables from a Dataset."""
        if isinstance(var, str):
            var = [var]
        try:
            return self.ds[var]
        except KeyError:
            raise KeyError(var)

    def align_time(self, reftimes: List[datetime], leadtimes: List[int]) -> xr.Dataset:
        """Select most recently available run, consider availability time"""
        ds = self.ds
        if len(ds) == 0:
            return ds
        if not "forecast_reference_time" in ds.dims:
            return ds
        ds = ds.assign_coords(init_time=ds.forecast_reference_time)
        if "arrival_time" in ds:
            ds["forecast_reference_time"] = ds.arrival_time.dt.floor("H")
        # take closest model run from the past (method="ffill")
        try:
            ds = ds.sortby("forecast_reference_time").sel(
                forecast_reference_time=reftimes,
                method="ffill",
            )
        except KeyError:
            arrival_times = ds.forecast_arrival_time.values
            raise ValueError(
                "model runs do not cover the requested time range"
                f"\n\treftime: ({reftimes[0]}, {reftimes[-1]})"
                f"\n\tmodel runs: ({arrival_times.min()}, {arrival_times.max()})"
            )

        # cast leadtimes to int
        new_leadtimes = ds["t"].astype("timedelta64[h]")
        new_leadtimes = new_leadtimes // np.timedelta64(1, "h")
        ds["t"] = new_leadtimes

        # Compute the time lag between model and reftimes
        lags_timedelta = (ds.forecast_reference_time - ds.init_time).astype(
            "timedelta64[h]"
        )
        lags_int = lags_timedelta // np.timedelta64(1, "h")
        lags_unique = sorted(list(set(lags_int.values)))

        new_ds = []
        for lag in lags_unique:
            iref = list(np.where(lags_int == lag)[0])
            ds_ = ds.isel(forecast_reference_time=iref)
            ds_ = ds_.interp(
                t=leadtimes + lag, method="nearest", kwargs={"fill_value": np.nan}
            )
            new_ds.append(ds_)

        # with warnings.catch_warnings():
        #     # Suppress PerformanceWarning
        #     warnings.simplefilter("ignore")
        new_ds = xr.concat(new_ds, dim="forecast_reference_time").sortby(
            "forecast_reference_time"
        )

        # Correct for static variables after concat
        for var in ds.data_vars:
            if not "forecast_reference_time" in ds[var].dims:
                new_ds[var] = new_ds[var].isel(forecast_reference_time=0, drop=True)
        # Finally update coordinates
        new_ds["forecast_reference_time"] = reftimes
        new_ds["t"] = leadtimes
        return new_ds.drop_vars(["arrival_time", "init_time"], errors="ignore")

    def unstack_time(
        self, reftimes: List[datetime], leadtimes: List[int]
    ) -> xr.Dataset:
        """
        Reshape a dataset from a linear time axis to a double time axis of
        reftime with leadtimes.
        """
        if not "time" in self.ds.dims:
            return self.ds
        reftimes = np.array(reftimes)
        leadtimes = np.array(leadtimes, dtype="timedelta64[h]")
        times = xr.DataArray(
            reftimes[:, None] + leadtimes,
            coords=[reftimes, leadtimes],
            dims=["forecast_reference_time", "t"],
        )
        times = times.where(times.isin(self.ds.time))
        times = times.dropna("forecast_reference_time", how="any")
        return self.ds.sel(time=times)

    def interp(self, points, **kwargs):
        """
        Interpolate all variables in the dataset onto a set of target points.
        """

        if "latitude" in self.ds:
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
