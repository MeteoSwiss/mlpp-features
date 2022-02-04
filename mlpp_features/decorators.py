from functools import wraps
import xarray as xr


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
        return out

    return inner
