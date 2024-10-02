import numpy as np
import xarray as xr

from mlpp_features import calc


def test_mean_sea_level_pressure_from_p_t_z():
    # Constants for the test
    g = 9.80665  # m/s²
    R = 287.05  # J/(kg·K)

    # Test input values
    p_surface = xr.DataArray([1013.25], dims=["x"])
    t_surface = xr.DataArray([15.0], dims=["x"])
    z_surface = xr.DataArray([100.0], dims=["x"])

    # Manually computed expected value using the formula
    expected_mslp = p_surface * np.exp((g * z_surface) / (R * (t_surface + 273.15)))

    # Call the function
    result = calc.mean_sea_level_pressure_from_p_t_z(p_surface, t_surface, z_surface)

    # Use pytest's assert for checking results
    np.testing.assert_allclose(result, expected_mslp, rtol=1e-6, atol=1e-2)
