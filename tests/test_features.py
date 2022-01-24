import numpy as np
import pytest

from mlpp_features import obs


NEIGHBOURS_RANKING_METHODS = [
    obs.variable_euclidean_nearest_k
    # obs.variable_some_metric_k
]


def test_variable_euclidean_nearest_k(raw_obs_dataset):

    k = 5

    obs_dataset = raw_obs_dataset()[["measurement"]].sel(variable="wind_speed")
    obs_nearest_k = obs.variable_euclidean_nearest_k(obs_dataset, k=k)

    good_dims = ["time", "point", "neighbour_rank"]
    assert list(obs_nearest_k.dims) == good_dims

    obs_at_aaa_original = obs_dataset.measurement.loc[:, 0].values
    obs_at_aaa_rank_zero = obs_nearest_k.measurement.loc[:, "AAA", 0].values
    np.testing.assert_equal(obs_at_aaa_original, obs_at_aaa_rank_zero)

    obs_at_rank_one_for_aaa_original = (
        obs_dataset.swap_dims({"station_id": "station_name"})
        .measurement.loc[:, obs_nearest_k.neighbour_name.loc["AAA", 1]]
        .values
    )

    obs_at_rank_one_for_aaa = obs_nearest_k.measurement.loc[:, "AAA", 1].values
    np.testing.assert_equal(obs_at_rank_one_for_aaa_original, obs_at_rank_one_for_aaa)


@pytest.mark.parametrize("ranking_method", NEIGHBOURS_RANKING_METHODS)
def test_variable_select_rank(raw_obs_dataset, ranking_method):

    k = 5
    rank = 0

    obs_dataset = raw_obs_dataset()[["measurement"]].sel(variable="wind_speed")
    obs_best_k = ranking_method(obs_dataset, k=k)
    obs_best_rank = obs.variable_select_rank(obs_best_k, rank=rank, k=k)

    good_dims = ["time", "point"]
    assert list(obs_best_rank.dims) == good_dims

    # check that missing samples are subsituted by a following ranked neighbour
    nan_idx = np.argwhere(np.isnan(obs_dataset.measurement.values))
    non_zero_rank_idx = np.argwhere(obs_best_rank.neighbour_rank.values != 0)
    assert np.all(nan_idx == non_zero_rank_idx)

    # check that the subsitute samples correspond to the original ones
    nan_idx = list(
        zip(
            obs_dataset.time[nan_idx[:, 0]].values,
            obs_dataset.station_name[nan_idx[:, 1]].values,
        )
    )
    nan_subs = obs_best_rank.stack(sample=["time", "point"]).sel(sample=nan_idx)
    nan_subs_idx = list(zip(nan_subs.time.values, nan_subs.neighbour_name.values))
    nan_sub_original = (
        obs_dataset.swap_dims({"station_id": "station_name"})
        .stack(sample=["time", "station_name"])
        .sel(sample=nan_subs_idx)
        .measurement
    )
    np.testing.assert_equal(nan_subs.values, nan_sub_original.values)
