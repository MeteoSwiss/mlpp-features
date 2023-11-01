import logging
import os
import pickle
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr

LOGGER = logging.getLogger(__name__)

KEEP_STA_COORDS = ["longitude", "latitude", "elevation", "owner_id"]


CACHEDIR_PREFIX = os.environ.get("MLPP_CACHE")
if CACHEDIR_PREFIX:
    LOGGER.info(f"Caching is enabled: {CACHEDIR_PREFIX}")
    Path(CACHEDIR_PREFIX).mkdir(parents=True, exist_ok=True)
    CACHEDIR = TemporaryDirectory(dir=CACHEDIR_PREFIX)


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

    To enable the cache, define the MLPP_CACHE environment variable pointing at the
    cache directory.

    Note: this works only if the arguments are always the same. This will usually
    be the case when extracting features, but beware of this caveat!
    """

    @wraps(fn)
    def inner(*args, **kwargs):
        if CACHEDIR_PREFIX is None:
            return fn(*args, **kwargs)
        cachedfile = Path(CACHEDIR.name) / f"{fn.__name__}.pkl"
        if not cachedfile.is_file():
            out = fn(*args, **kwargs).compute()
            LOGGER.debug(f"to cache: {cachedfile}")
            with open(cachedfile, "wb") as f:
                pickle.dump(out, f, protocol=-1)
        else:
            LOGGER.debug(f"from cache: {cachedfile}")
            with open(cachedfile, "rb") as f:
                out = pickle.load(f)
        return out.chunk("auto").persist()

    return inner
