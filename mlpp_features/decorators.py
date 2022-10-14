from functools import wraps
import xarray as xr
import pickle
import pathlib
from itertools import chain

KEEP_STA_COORDS = ["longitude", "latitude", "elevation", "owner_id"]


def io(
    dependencies: dict[str, list],
    endpoint: bool = True,
    units: str = None,
    cache: bool = False,
):
    def decorator(fn):
        def wrapped(*args, **kwargs):

            # check dependencies
            try:
                for source, deps in dependencies.items():
                    if deps == []:
                        continue
                    for dep in deps:
                        args[0][source][dep]
            except KeyError:
                alldeps = list(
                    chain.from_iterable([dep for dep in dependencies.values()])
                )
                raise KeyError(alldeps)

            # compute pipeline
            if cache:
                cachedfile = pathlib.Path(kwargs["tmp_dir"]) / f"{fn.__name__}"
                if not cachedfile.is_file():
                    out = fn(*args, **kwargs).load()
                    with open(cachedfile, "wb") as f:
                        pickle.dump(out, f, protocol=-1)
                else:
                    with open(cachedfile, "rb") as f:
                        out = pickle.load(f)
            else:
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

        wrapped.__endpoint__ = endpoint

        return wrapped

    return decorator


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
