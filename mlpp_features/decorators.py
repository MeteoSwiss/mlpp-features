from functools import wraps
import xarray as xr


KEEP_STA_COORDS = ["longitude", "latitude", "elevation", "owner_id"]


def asarray(func):
    """
    Make every feature function return a DataArray names like the function itself.
    """

    @wraps(func)
    def inner(*args, **kwargs):
        out = func(*args, **kwargs)
        if isinstance(out, xr.Dataset):
            out = out.to_array(name=func.__name__).squeeze("variable", drop=True)
        elif isinstance(out, xr.DataArray):
            out = out.rename(func.__name__)
        # cleanup coordinates
        out = out.drop_vars(
            [c for c in out.coords if c not in list(out.dims) + KEEP_STA_COORDS]
        )
        return out

    return inner


def reuse(func):
    """
    If the feature already exists in the target dataset, reuse it instead of computing it again.
    The target dataset is where the pipelines' results are accumulated, and can be passed
    as a keyword argument named `ds`.
    """

    @wraps(func)
    def inner(*args, **kwargs):
        ds = kwargs.get("ds", xr.Dataset())
        if func.__name__ in ds.data_vars:
            out = ds[func.__name__]
        else:
            out = func(*args, **kwargs)
        return out

    return inner
