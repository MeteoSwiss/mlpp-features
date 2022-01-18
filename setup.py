# minimal setup.y until poetry supports PEP660
# https://github.com/python-poetry/poetry/issues/34
from setuptools import setup, find_packages

requirements = [
    "zarr",
    "dask",
    "xarray",
    "pyproj",
    "scipy",
]

setup(
    name="mlpp-features",
    install_requires=requirements,
    packages=find_packages(include=["mlpp_features"]),
)
