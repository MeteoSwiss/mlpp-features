# MLPP Features

[![.github/workflows/run-tests.yml](https://github.com/MeteoSwiss/mlpp-features/actions/workflows/run-tests.yml/badge.svg)](https://github.com/MeteoSwiss/mlpp-features/actions/workflows/run-tests.yml)
[![pypi](https://img.shields.io/pypi/v/mlpp-features.svg?colorB=<brightgreen>)](https://pypi.python.org/pypi/mlpp-features/)

Define, track, share, and discover features for ML-based postprocessing of weather forecasts. 

:warning: **The code in this repository is currently work-in-progress and not recommended for production use.** :warning:

## Installation
Clone the `mlpp-features` repository  
``` 
git clone git@github.com:MeteoSwiss/mlpp-features.git
cd mlpp-features
```
Install the project via  
```
pip install -e .
```

## Getting started
The repo is organised mainly in three parts.  
Features are defined in [`nwp.py`](https://github.com/MeteoSwiss/mlpp-features/blob/main/mlpp_features/nwp.py), [`obs.py`](https://github.com/MeteoSwiss/mlpp-features/blob/main/mlpp_features/obs.py), [`terrain.py`](https://github.com/MeteoSwiss/mlpp-features/blob/main/mlpp_features/terrain.py) and [`time.py`](https://github.com/MeteoSwiss/mlpp-features/blob/main/mlpp_features/time.py) depending on their nature.  
Some features are not natively produced by NWP models but derived from other variables. Relevant function are defined in the [`calc`](https://github.com/MeteoSwiss/mlpp-features/blob/main/mlpp_features/calc.py) module. Geographical calculations are handled in the [`geo_calculations.py`](https://github.com/MeteoSwiss/mlpp-features/blob/main/mlpp_features/geo_calculations.py) module.  
Finally, [`accessors.py`](https://github.com/MeteoSwiss/mlpp-features/blob/main/mlpp_features/accessors.py) and [`selectors.py`](https://github.com/MeteoSwiss/mlpp-features/blob/main/mlpp_features/selectors.py) are used to define utility functions.
