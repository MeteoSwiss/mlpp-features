import mlpp_features.experimental as exp
import numpy as np


def test_sign_distance_alpine_ridge(alpine_ridge, diagonal_ridge, horizontal_ridge):

    for ridge in [alpine_ridge, diagonal_ridge, horizontal_ridge]:
        test_stations_north = [(lat+0.5, lon) for lat, lon in ridge]
        test_stations_south = [(lat-0.5, lon) for lat, lon in ridge]

        assert all(
            np.sign(exp.distances_points_to_line(test_stations_north, ridge)) == 1.
        ), f"{np.sign(exp.distances_points_to_line(test_stations_north, ridge))}, {ridge[0]}"
        
        assert all(
            np.sign(exp.distances_points_to_line(test_stations_south, ridge)) == -1.
        ), f"{np.sign(exp.distances_points_to_line(test_stations_south, ridge))}"

        assert all(
            np.sign(exp.distances_points_to_line(ridge, ridge)) == 0.
        ), f"{np.sign(exp.distances_points_to_line(ridge, ridge))}"


def test_distance_horizontal_ridge(horizontal_ridge):

    test_stations_on_ridge = [(horizontal_ridge[0][0], float(i)) for i in range(6, 11)]
    test_stations_north = [(horizontal_ridge[0][0]+0.5, float(i)) for i in range(6, 11)]

    d = exp.distances_points_to_line(test_stations_on_ridge, horizontal_ridge)
    assert np.allclose(d, 0.), f"{d}"