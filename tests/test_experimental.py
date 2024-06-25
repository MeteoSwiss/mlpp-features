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
    test_stations_south = [(horizontal_ridge[0][0]-0.5, float(i)) for i in range(6, 11)]

    d_on_ridge = exp.distances_points_to_line(test_stations_on_ridge, horizontal_ridge)
    assert np.allclose(d_on_ridge, 0.), f"{d_on_ridge}"

    d_north = exp.distances_points_to_line(test_stations_north, horizontal_ridge)
    assert np.all(d_north > 0.), f"{d_north}"
    assert np.allclose(d_north, 55.59746332), f"{d_north}"

    d_south = exp.distances_points_to_line(test_stations_south, horizontal_ridge)
    assert np.all(d_south < 0.), f"{d_south}"
    assert np.allclose(d_south, -55.59746332), f"{d_south}"


def test_distance_diagonal_ridge(diagonal_ridge):

    test_stations_on_ridge = [(45.8, 7.0), (46.1, 8.0), (46.4, 9.0), (46.7, 10.0)]
    test_stations_north = [(lat+0.5, lon) for lat, lon in test_stations_on_ridge]
    test_stations_south = [(lat-0.5, lon) for lat, lon in test_stations_on_ridge]

    d_on_ridge = exp.distances_points_to_line(test_stations_on_ridge, diagonal_ridge)
    assert np.allclose(d_on_ridge, 0.), f"{d_on_ridge}"

    d_north = exp.distances_points_to_line(test_stations_north, diagonal_ridge)
    assert np.all(d_north > 0.), f"{d_north}"
    assert np.allclose(d_north, [52.09988376, 52.08812697, 52.07637342, 52.06462441]), f"{d_north}"

    d_south = exp.distances_points_to_line(test_stations_south, diagonal_ridge)
    assert np.all(d_south < 0.), f"{d_south}"
    assert np.allclose(d_south, [-52.12110037, -52.109341, -52.09758254, -52.08582629]), f"{d_south}"


def test_distance_alpine_ridge(alpine_ridge):

    test_stations_on_ridge = [(lat, lon) for lat, lon in alpine_ridge]
    test_stations_north = [(lat+0.5, lon) for lat, lon in alpine_ridge]
    test_stations_south = [(lat-0.5, lon) for lat, lon in alpine_ridge]

    d_on_ridge = exp.distances_points_to_line(test_stations_on_ridge, alpine_ridge)
    assert np.allclose(d_on_ridge, 0.), f"{d_on_ridge}"

    d_north = exp.distances_points_to_line(test_stations_north, alpine_ridge)
    assert np.all(d_north > 0.), f"{d_north}"
    assert np.allclose(d_north, [35.62369523, 45.38753682, 54.55866622, 48.59926533, 41.30797226,
                                 47.72705983, 55.58869808, 51.55969523, 55.59746332, 45.66260442,
                                 51.94482634, 51.80261932, 36.41012219, 37.5048918,  55.0139833,
                                 55.59746332]), f"{d_north}"
    
    d_south = exp.distances_points_to_line(test_stations_south, alpine_ridge)
    assert np.all(d_south < 0.), f"{d_south}"
    assert np.allclose(d_south, [-55.59746332, -47.9938393,  -35.68073047, -41.15784382, -48.61454682,
                                 -41.47559454, -45.33893686, -55.44646677, -45.71482408, -55.59746332,
                                 -51.9394463,  -55.59746332, -55.5593951,  -36.51986713, -36.43707536,
                                 -55.0168659]), f"{d_south}"
