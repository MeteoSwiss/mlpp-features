import pytest

import mlpp_features.discover as di


def test_discover_inputs():

    vars = di.discover_inputs("model_height_difference")
    assert vars == ["DEM", "surface_altitude"]

    vars = di.discover_inputs(
        ["air_temperature_ensctrl", "water_vapor_mixing_ratio_ensavg"]
    )
    assert vars == ["air_temperature", "dew_point_temperature", "surface_air_pressure"]

    vars = di.discover_inputs("wind_speed_ensavg")
    assert vars == ["eastward_wind", "northward_wind"]

    vars = di.discover_inputs(["aspect_500m", "wind_speed_of_gust_ensavg"])
    assert vars == [
        "ASPECT_500M_SIGRATIO1",
        "wind_speed_of_gust",
    ]  # sorted alphabetically

    vars = di.discover_inputs(["wind_speed", "nearest_wind_speed_of_gust"])
    assert vars == ["wind_speed", "wind_speed_of_gust"]  # observations

    vars = di.discover_inputs(["water_vapor_mixing_ratio"])
    assert vars == ["air_temperature", "relative_humidity", "surface_air_pressure"]

    vars = di.discover_inputs("weight_owner_id")
    assert vars == []

    vars = di.discover_inputs("cos_valley_index_1000m")
    assert vars == ["VALLEY_DIR_1000M_SMTHFACT0.5", "VALLEY_NORM_1000M_SMTHFACT0.5"]

    vars = di.discover_inputs(
        [
            "air_temperature_ensavg",
            "cos_dayofyear",
            "cos_hourofday",
            "boundary_layer_height_ensavg",
            "pressure_difference_BAS_LUG_ensavg",
            "pressure_difference_GVE_GUT_ensavg",
            "wind_speed_of_gust_ensavg",
            "wind_speed_of_gust_ensstd",
        ]
    )
    assert vars == [
        "air_temperature",
        "atmosphere_boundary_layer_thickness",
        "surface_air_pressure",
        "wind_speed_of_gust",
    ]
