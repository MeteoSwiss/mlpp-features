from datetime import timedelta

import pandas as pd
import xarray as xr

import mlpp_features  # type: ignore


def test_align_time(preproc_dataset):
    time_shift = 1
    n_reftimes = 3

    ds = preproc_dataset()

    t0 = pd.Timestamp(ds.forecast_reference_time.values[0])
    reftimes = [t0 + (1 + n) * timedelta(hours=time_shift) for n in range(n_reftimes)]
    reftimes = pd.DatetimeIndex(reftimes)
    leadtimes = [0, 1]
    ds_aligned = ds.preproc.align_time(reftimes, leadtimes)
    assert isinstance(ds_aligned, xr.Dataset)
    assert ds_aligned.sizes["forecast_reference_time"] == len(reftimes)
    assert ds_aligned.sizes["t"] == len(leadtimes)
    ds_aligned.sel(forecast_reference_time=reftimes)
    ds_aligned.sel(t=leadtimes)

    for var, da in ds.items():
        array_aligned = ds_aligned.isel(forecast_reference_time=0).sel(t=0)[var].values
        array_original = da.isel(
            t=time_shift, forecast_reference_time=0, missing_dims="ignore"
        ).values
        assert (array_aligned == array_original).all()
