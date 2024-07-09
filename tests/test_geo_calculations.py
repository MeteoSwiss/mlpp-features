import mlpp_features.geo_calculations as geo
import numpy as np


def test_sign_distance_alpine_ridge(alpine_ridge, diagonal_ridge, horizontal_ridge):

    for ridge in [diagonal_ridge, horizontal_ridge]:
        test_stations_north = [(lat + 0.005, lon) for lat, lon in ridge]
        test_stations_south = [(lat - 0.005, lon) for lat, lon in ridge]

        assert all(
            np.sign(geo.distances_points_to_line(test_stations_north, ridge)) == 1.0
        ), f"{np.sign(geo.distances_points_to_line(test_stations_north, ridge))}, {ridge[0]}"

        assert all(
            np.sign(geo.distances_points_to_line(test_stations_south, ridge)) == -1.0
        ), f"{np.sign(geo.distances_points_to_line(test_stations_south, ridge))}"

        assert all(
            np.sign(geo.distances_points_to_line(ridge, ridge)) == 0.0
        ), f"{np.sign(geo.distances_points_to_line(ridge, ridge))}"

    test_stations_alpine_ridge = [
        (45.65975, 6.88306),
        (48.0, 7.0),
        (48.0, 11.3),
        (45.6, 11.3),
        (45.6, 6.5),
    ]
    assert all(
        np.sign(geo.distances_points_to_line(test_stations_alpine_ridge, alpine_ridge))
        == [-1, 1, 1, -1, -1]
    ), f"{np.sign(geo.distances_points_to_line(test_stations_alpine_ridge, alpine_ridge))} != [-1, 1, 1, -1, -1]"


def test_distance_horizontal_ridge(horizontal_ridge):

    test_stations_on_ridge = [(horizontal_ridge[0][0], float(i)) for i in range(6, 11)]
    test_stations_north = [
        (horizontal_ridge[0][0] + 0.5, float(i)) for i in range(6, 11)
    ]
    test_stations_south = [
        (horizontal_ridge[0][0] - 0.5, float(i)) for i in range(6, 11)
    ]

    d_on_ridge = geo.distances_points_to_line(test_stations_on_ridge, horizontal_ridge)
    assert np.allclose(d_on_ridge, 0.0), f"{d_on_ridge}"

    d_north = geo.distances_points_to_line(test_stations_north, horizontal_ridge)
    assert np.all(d_north > 0.0), f"{d_north}"
    assert np.allclose(d_north, 55.59746332), f"{d_north}"

    d_south = geo.distances_points_to_line(test_stations_south, horizontal_ridge)
    assert np.all(d_south < 0.0), f"{d_south}"
    assert np.allclose(d_south, -55.59746332), f"{d_south}"


def test_distance_diagonal_ridge(diagonal_ridge):

    test_stations_on_ridge = [(45.8, 7.0), (46.1, 8.0), (46.4, 9.0), (46.7, 10.0)]
    test_stations_north = [(lat + 0.5, lon) for lat, lon in test_stations_on_ridge]
    test_stations_south = [(lat - 0.5, lon) for lat, lon in test_stations_on_ridge]

    d_on_ridge = geo.distances_points_to_line(test_stations_on_ridge, diagonal_ridge)
    assert np.allclose(d_on_ridge, 0.0), f"{d_on_ridge}"

    d_north = geo.distances_points_to_line(test_stations_north, diagonal_ridge)
    assert np.all(d_north > 0.0), f"{d_north}"
    assert np.allclose(
        d_north, [52.09988376, 52.08812697, 52.07637342, 52.06462441]
    ), f"{d_north}"

    d_south = geo.distances_points_to_line(test_stations_south, diagonal_ridge)
    assert np.all(d_south < 0.0), f"{d_south}"
    assert np.allclose(
        d_south, [-52.12110037, -52.109341, -52.09758254, -52.08582629]
    ), f"{d_south}"


def test_distance_alpine_ridge(alpine_ridge):

    test_stations_on_ridge = [(lat, lon) for lat, lon in alpine_ridge]
    test_stations = [
        (45.7, 7.0),
        (46.7, 8.96059),
        (46.4, 8.96059),
        (45.9, 7.07724),
        (45.8, 7.07724),
    ]

    d_on_ridge = geo.distances_points_to_line(test_stations_on_ridge, alpine_ridge)
    assert np.allclose(d_on_ridge, 0.0), f"{d_on_ridge}"

    d = geo.distances_points_to_line(test_stations, alpine_ridge)
    assert d[0] < 0.0 and d[2] < 0.0 and d[4] < 0.0, f"{d[0]}, {d[2]}, {d[4]}"
    assert d[1] > 0.0 and d[3] > 0.0, f"{d[1]}, {d[3]}"
