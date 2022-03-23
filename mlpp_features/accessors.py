""""""
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

import mlpp_features.selectors as sel


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
    selector: sel.StationSelector = field(init=False, repr=False)

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
        if not "forecast_reference_time" in ds.dims:
            return ds

        # Rearrange coordinates
        ds = ds.assign_coords(init_time=ds.forecast_reference_time)
        if "arrival_time" in ds:
            ds["forecast_reference_time"] = ds.arrival_time

        # take closest model run from the past (method="ffill")
        try:
            ds = ds.sortby("forecast_reference_time").sel(
                forecast_reference_time=reftimes,
                method="ffill",
            )
        except KeyError:
            arrival_times = ds.forecast_reference_time.values
            raise ValueError(
                "model runs do not cover the requested time range"
                f"\n\treftime: ({reftimes[0]}, {reftimes[-1]})"
                f"\n\tmodel runs: ({arrival_times.min()}, {arrival_times.max()})"
            )

        #  Cast leadtimes to int
        new_leadtimes = ds["t"].astype("timedelta64[h]")
        new_leadtimes = new_leadtimes // np.timedelta64(1, "h")
        ds["t"] = new_leadtimes

        # Compute the time lag between model and reftimes
        reftimes = np.array(list(map(np.datetime64, reftimes)))
        lags_timedelta = (reftimes - ds.init_time).astype("timedelta64[h]")
        lags_int = lags_timedelta // np.timedelta64(1, "h")
        lags_unique = sorted(list(set(lags_int.values)))

        new_ds = []
        for lag in lags_unique:
            iref = list(np.where(lags_int == lag)[0])
            ds_ = ds.isel(forecast_reference_time=iref)
            ds_ = ds_.interp(
                t=leadtimes + lag, method="nearest", kwargs={"fill_value": np.nan}
            )
            ds_["t"] = leadtimes
            ds_ = ds_.assign_coords({"leadtime": ("t", leadtimes + lag)})
            new_ds.append(ds_)

        with warnings.catch_warnings():
            # Suppress PerformanceWarning
            warnings.simplefilter("ignore")
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
        new_ds = new_ds.drop_vars(["arrival_time", "init_time"], errors="ignore")

        return new_ds

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
        leadtimes_td = np.array(leadtimes, dtype="timedelta64[h]")
        times = xr.DataArray(
            reftimes[:, None] + leadtimes_td,
            coords=[reftimes, leadtimes],
            dims=["forecast_reference_time", "t"],
        )
        times = times.where(times.isin(self.ds.time))
        times = times.dropna("forecast_reference_time", how="any")
        new_ds = self.ds.sel(time=times)
        return new_ds.drop_vars("time", errors="ignore")

    def persist_observations(
        self, reftimes: List[datetime], leadtimes: List[int]
    ) -> xr.Dataset:
        """Persist the latest observation to all leadtimes."""
        if not "time" in self.ds.dims:
            return self.ds
        reftimes = np.array(reftimes)
        leadtimes_td = np.array(leadtimes, dtype="timedelta64[h]")
        reftimes = reftimes[np.isin(reftimes, self.ds.time.values)]
        times = xr.DataArray(
            reftimes[:, None] + leadtimes_td,
            coords=[reftimes, leadtimes],
            dims=["forecast_reference_time", "t"],
        )
        ds = self.ds.sel(time=reftimes).rename({"time": "forecast_reference_time"})
        ds = ds.expand_dims(t=leadtimes, axis=1).assign_coords(time=times)
        return ds

    def interp(self, stations: pd.DataFrame, **kwargs):
        """
        Interpolate all variables in the dataset onto a set of target stations.
        """
        if "latitude" in self.ds:
            selector = sel.EuclideanNearestIrregular(self.ds)
        else:
            selector = sel.EuclideanNearestRegular(self.ds)
        index = selector.query(stations, **kwargs)
        stations = stations[index.valid.to_series()]
        index = index.where(index.valid, drop=True).astype(int)
        ds_out = (
            self.ds.stack(point=("y", "x"))
            .isel(point=index)
            .drop_vars(("point", "valid", "distance"))
            .assign_coords({c: ("station", v.values) for c, v in stations.iteritems()})
        )
        return ds_out

    def euclidean_nearest_k(self, stations: pd.DataFrame, k: int) -> xr.Dataset:
        """
        Select k nearest neighbors using euclidean distance.
        """
        selector = sel.EuclideanNearestSparse(self.ds)
        index = selector.query(stations, k=k)
        ds = self.ds.rename({"station": "neighbor"})
        ds = ds.isel(neighbor=index)
        ds = ds.rename(
            {
                "longitude": "neighbor_longitude",
                "latitude": "neighbor_latitude",
                "elevation": "neighbor_elevation",
                "distance": "neighbor_distance",
            }
        )
        ds = ds.assign_coords(
            {c: ("station", v.values) for c, v in stations.iteritems()}
        )
        return ds

    def select_rank(self, rank: int) -> xr.Dataset:
        """
        Select the ranked observations at each timestep.
        """
        k = len(self.ds.neighbor_rank.values)
        if k <= rank:
            raise ValueError(f"not enough neighbors to extract rank {rank}")
        if len(self.ds) > 1:
            raise ValueError(f"can only select rank for datasets of 1 variable")
        ds_out = xr.Dataset()
        for var, da in self.ds.data_vars.items():
            da = da.sel(neighbor_rank=slice(rank, None))
            # find index of nearest non-missing measurement at each time
            mask = np.isfinite(da)
            index = xr.where(
                mask.any(dim="neighbor_rank"), mask.argmax(dim="neighbor_rank"), -1
            )
            # make sure no index is > k
            index = xr.where(index <= k, index, k - 1)
            # indexing with chunked arrays fails
            index.load()
            ds_out[var] = da.isel(neighbor_rank=index)
        for co in ds_out.coords:
            if "neighbor_" not in co:
                continue
            ds_out = ds_out.rename({co: co.replace("neighbor_", f"neighbor_{rank}_")})
        return ds_out.transpose("time", "station")

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

    def circmean(self, dim):
        """Compute the mean of an array of wind directions between 0 and 360 degrees"""
        sin_mean = np.sin(self.ds * np.pi / 180).mean(dim)
        cos_mean = np.cos(self.ds * np.pi / 180).mean(dim)
        wdir = np.arctan2(sin_mean, cos_mean) * 180 / np.pi + 360
        return wdir % 36

    def wind_from_direction(self):
        """
        Compute the meteorological wind direction (i.e., the direction from which the
        wind is blowing) of a wind vector defined by its zonal and meridional components,
        that is, its eastward and northward velocities.
        """
        u = self.ds["eastward_wind"]
        v = self.ds["northward_wind"]
        da = np.arctan2(v, u)
        da = (270 - 180 / np.pi * da) % 360
        da.attrs["units"] = "degrees"
        return da.to_dataset(name="wind_from_direction")
