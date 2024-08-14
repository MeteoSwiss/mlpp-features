""""""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Union, Callable

import numpy as np
import pandas as pd
import xarray as xr
from bottleneck import rankdata as bn_rankdata

import mlpp_features.selectors as sel

LOGGER = logging.getLogger(__name__)

# Set global options
xr.set_options(keep_attrs=True)

# Silence non-nanosecond conversion warning
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Converting non-nanosecond.*"
)


@xr.register_dataset_accessor("mlpp")
@dataclass
class PreprocDatasetAccessor:
    """
    Access methods for Datasets with mlpp preprocessing methods.
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

    def align_time(
        self,
        reftimes: Union[List[datetime], None],
        leadtimes: Union[List[timedelta], None],
        return_source_leadtimes: bool = False,
    ) -> xr.Dataset:
        """Select most recently available run, consider availability time"""

        ds = self.ds
        if not "forecast_reference_time" in ds.dims:
            return ds

        if reftimes is None or leadtimes is None:
            return ds

        reftimes = np.array(reftimes, dtype="datetime64[h]")
        leadtimes = np.array(leadtimes, dtype="timedelta64[h]")

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

        # Compute the time lag between model and reftimes
        lags_timedelta = np.array(reftimes - ds.init_time, dtype="timedelta64[h]")
        lags_unique = np.unique(lags_timedelta)

        new_ds = []
        for lag in lags_unique:
            iref = list(np.where(lags_timedelta == lag)[0])
            ds_sub = ds.isel(forecast_reference_time=iref)
            lagged_leadtimes = (leadtimes + lag).astype("timedelta64[ns]")
            # pad to make sure that out-of-range leadtimes are filled with nans
            ds_sub = ds_sub.pad(
                {"lead_time": (0, 1)}, mode="constant", constant_values=np.nan
            )
            ds_sub["lead_time"] = np.append(
                ds_sub["lead_time"].values[:-1],
                ds_sub["lead_time"].values[-2] + np.timedelta64(1, "s"),
            )
            ds_sub = ds_sub.sel(
                lead_time=lagged_leadtimes,
                method="nearest",
            )
            ds_sub["lead_time"] = leadtimes.astype("timedelta64[ns]")
            if return_source_leadtimes:
                ds_sub = ds_sub.assign_coords(
                    {"source_leadtime": ("lead_time", lagged_leadtimes)}
                )
            new_ds.append(ds_sub)

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
        new_ds["forecast_reference_time"] = reftimes.astype("datetime64[ns]")
        new_ds["lead_time"] = leadtimes.astype("timedelta64[ns]")
        new_ds = new_ds.drop_vars(["arrival_time", "init_time"], errors="ignore")

        if (
            return_source_leadtimes
            and "forecast_reference_time" not in new_ds["source_leadtime"].dims
        ):
            new_ds["source_leadtime"] = new_ds["source_leadtime"].expand_dims(
                forecast_reference_time=reftimes
            )

        return new_ds

    def unstack_time(
        self, reftimes: List[datetime], leadtimes: List[timedelta]
    ) -> xr.Dataset:
        """
        Reshape a dataset from a linear time axis to a double time axis of
        reftime with leadtimes.
        """
        if not "time" in self.ds.dims:
            return self.ds

        reftimes = np.array(reftimes, dtype="datetime64[ns]")
        leadtimes = np.array(leadtimes, dtype="timedelta64[ns]")

        times = xr.DataArray(
            reftimes[:, None] + leadtimes,
            coords=[reftimes, leadtimes],
            dims=["forecast_reference_time", "lead_time"],
        )
        times = times.where(times.isin(self.ds.time))
        times = times.dropna("forecast_reference_time", how="any")
        new_ds = self.ds.sel(time=times)
        return new_ds.drop_vars("time", errors="ignore")

    def persist_observations(
        self, reftimes: List[datetime], leadtimes: List[timedelta]
    ) -> xr.Dataset:
        """Persist the latest observation to all leadtimes."""
        if not "time" in self.ds.dims:
            return self.ds

        reftimes = np.array(reftimes, dtype="datetime64[ns]")
        leadtimes = np.array(leadtimes, dtype="timedelta64[ns]")

        reftimes = reftimes[np.isin(reftimes, self.ds.time.values)]
        times = xr.DataArray(
            reftimes[:, None] + leadtimes,
            coords=[reftimes, leadtimes],
            dims=["forecast_reference_time", "lead_time"],
        )
        ds = self.ds.sel(time=reftimes).rename({"time": "forecast_reference_time"})
        ds = ds.expand_dims(lead_time=leadtimes, axis=1).assign_coords(time=times)
        return ds

    def interp(self, stations: pd.DataFrame, **kwargs):
        """
        Interpolate all variables in the dataset onto a set of target stations.
        """
        if "latitude" in self.ds:
            selector = sel.EuclideanNearestIrregular(self.ds)
        else:
            selector = sel.EuclideanNearestRegular(self.ds)
        valid_arguments = ["search_radius", "vertical_weight"]
        query_kwargs = {k: v for k, v in kwargs.items() if k in valid_arguments}
        index = selector.query(stations, **query_kwargs)
        stations = stations[index.valid.to_series()]
        index = index.where(index.valid, drop=True).astype(int)
        ds_out = (
            self.ds.stack(point=("y", "x"))
            .isel(point=index.sortby(index))
            .drop_vars(("point", "valid", "distance"))
            .compute()
            .reindex(station=list(stations.index))
            .assign_coords({c: ("station", v.values) for c, v in stations.items()})
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
                "height_masl": "neighbor_height_masl",
                "distance": "neighbor_distance",
            }
        )
        ds = ds.assign_coords({c: ("station", v.values) for c, v in stations.items()})
        return ds

    def daystat(self, func: Callable, complete: bool = True) -> xr.Dataset:
        """
        Compute daily summaries on xr.Dataset.
        """
        ds = self.ds

        # follow "end-of-accumulation time" convention
        ds["time"] = ds.time - np.timedelta64(1, "h")
        res = []
        for reftime in ds.forecast_reference_time:
            ds_tmp = ds.sel(forecast_reference_time=reftime)
            res_reftime = []
            for i, group in ds_tmp.groupby("time.date"):
                dayfunc = func(group, dim="lead_time", skipna=~complete)
                dayfunc = dayfunc.broadcast_like(group).unstack()
                res_reftime.append(dayfunc)
            res.append(xr.merge(res_reftime))
        res = xr.concat(res, "forecast_reference_time")
        res.coords["time"] = ds["time"] + np.timedelta64(1, "h")
        return res

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
        return wdir % 360

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

    def rankdata(
        self,
        dim: str = "realization",
        loop: Optional[str] = None,
        circular: bool = False,
    ) -> xr.Dataset:
        """
        Perform ranking along one dimension with the option of looping along
        a second dimension to reduce memory usage. Currently missing data
        are not supported.

        Parameters
        ----------
        dim: str
            Dimension along which to perform the ranking.
        loop: str, optional
            If specified, ranking is done while looping along the given dimension
            to limit memory.
        circular: bool
            Compute ranks of circular data, i.e. angles between 0 and 360 degrees.
        """

        if loop is None:
            loop = "dummy"

        output = xr.Dataset()
        for var, da in self.ds.data_vars.items():
            if dim not in da.dims:
                continue

            if circular:
                sin_sum = np.sin(da * np.pi / 180).sum(dim)
                cos_sum = np.cos(da * np.pi / 180).sum(dim)
                da_mean = np.arctan2(sin_sum, cos_sum) * 180 / np.pi + 360
                rank_origin = (da_mean + 180) % 360
                da = da.where(da >= rank_origin, da + 360)

            # resolve ties at random
            da += np.random.random(da.shape) / 1e10

            if loop == "dummy":
                da = da.expand_dims("dummy")
            dai = []
            for i in range(da[loop].size):
                rank_kwargs = {"axis": list(da.dims).index(dim)}
                dai.append(
                    xr.apply_ufunc(
                        bn_rankdata,
                        da.isel({loop: slice(i, i + 1)}),
                        kwargs=rank_kwargs,
                        dask="parallelized",
                    ).astype("float32")
                )
            da = xr.concat(dai, loop, join="override")
            da.attrs["units"] = "rank"
            if loop == "dummy":
                da = da.squeeze("dummy", drop=True)
            output[var] = da
        return output
