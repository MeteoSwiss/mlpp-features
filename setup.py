# minimal setup.y until poetry supports PEP660
# https://github.com/python-poetry/poetry/issues/34
from setuptools import setup

requirements = [
    "zarr",
    "dask",
    "xarray",
    "pyproj",
    "scipy",
]

setup(
    name="mlpp-features",
    packages=["mlpp_features"],
    install_requires=requirements,
)
