from functools import wraps
from importlib.resources import path
import xarray as xr
import pickle
from pathlib import Path
from itertools import chain
import os
import tempfile

KEEP_STA_COORDS = ["longitude", "latitude", "elevation", "owner_id"]


CACHEDIR_PREFIX = Path(os.environ.get("MLPP_CACHE", f"{os.getcwd()}/mlpp_cache/"))
CACHEDIR_PREFIX.mkdir(parents=True, exist_ok=True)
CACHEDIR = tempfile.TemporaryDirectory(prefix=f"{CACHEDIR_PREFIX.as_posix()}/")


def out_format(
    units: str = None,
):
    """
    Additional formatting of a pipeline's output:
        - transforms it to a `xr.DataArray` object named like the function itself
        - adds the `units` attribute
    """

    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):

            out = fn(*args, **kwargs)

            # return as array with fn name
            if isinstance(out, xr.Dataset):
                out = out.to_array(name=fn.__name__).squeeze("variable", drop=True)
            elif isinstance(out, xr.DataArray):
                out = out.rename(fn.__name__)

            # drop coordinates
            out = out.drop_vars(
                [c for c in out.coords if c not in list(out.dims) + KEEP_STA_COORDS]
            )

            if units is not None:
                out.attrs["units"] = units

            return out

        return wrapped

    return decorator


def cache(fn):
    """
    Cache results on disk as pickle objects the first time `fn` is called,
    then loads them back when needed again.

    Note: this works only if the arguments are always the same. This will usually
    be the case when extracting features, but beware of this caveat!
    """

    @wraps(fn)
    def inner(*args, **kwargs):

        cachedfile = Path(CACHEDIR.name) / f"{fn.__name__}.pkl"
        if not cachedfile.is_file():
            out = fn(*args, **kwargs).compute()
            with open(cachedfile, "wb") as f:
                pickle.dump(out, f, protocol=-1)
        else:
            with open(cachedfile, "rb") as f:
                out = pickle.load(f)
        return out.chunk("auto").persist()

    return inner
