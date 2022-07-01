from typing import List, Union

import xarray as xr

import mlpp_features  # type: ignore


def discover_inputs(pipelines: Union[str, List[str]]) -> List:
    """
    Return a sorted and unique list with the names of all the input data that are
    required for the predictor's pipelines.
    """
    if isinstance(pipelines, str):
        pipelines = [pipelines]

    # pass an empty dataset to trigger a KeyError
    data = {"nwp": xr.Dataset(), "terrain": xr.Dataset(), "obs": xr.Dataset()}
    inputs = []
    for pipeline in pipelines:
        try:
            getattr(globals()["mlpp_features"], pipeline)(data, None, None, None)
        except KeyError as err:
            var = err.args[0]
            if isinstance(var, str):
                inputs.append(var)
            else:
                inputs += var
    return sorted(list(set(inputs)))
