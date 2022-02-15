import mlpp_features.discover as di


def test_discover_inputs():

    vars = di.discover_inputs("wind_speed_ensavg")
    assert vars == ["eastward_wind", "northward_wind"]

    vars = di.discover_inputs(["aspect_500m", "wind_speed_of_gust_ensavg"])
    assert vars == [
        "ASPECT_500M_SIGRATIO1",
        "wind_speed_of_gust",
    ]  # sorted alphabetically

    vars = di.discover_inputs(["wind_speed", "nearest_wind_speed_of_gust"])
    assert vars == ["wind_speed", "wind_speed_of_gust"]  # observations
