import numpy as np

import mlpp_features  # type: ignore


def stations_df_from_obs_ds(obs_dataset):
    stations = obs_dataset[
        ["station", "longitude", "latitude", "elevation"]
    ].to_pandas()
    return stations


def test_variable_euclidean_nearest_k(raw_obs_dataset):

    k = 5
    obs_dataset = raw_obs_dataset()[["wind_speed"]]
    stations = stations_df_from_obs_ds(obs_dataset)
    obs_nearest_k = obs_dataset.preproc.euclidean_nearest_k(stations, k)
    assert list(obs_nearest_k.dims) == ["time", "station", "neighbor_rank"]

    obs_at_klo_original = obs_dataset.wind_speed.sel(station="KLO").values
    obs_at_klo_rank_zero = obs_nearest_k.wind_speed.sel(
        station="KLO", neighbor_rank=0
    ).values
    np.testing.assert_equal(obs_at_klo_original, obs_at_klo_rank_zero)

    obs_at_rank_one_for_klo_original = obs_dataset.wind_speed.loc[
        :, obs_nearest_k.neighbor.loc["KLO", 1]
    ].values
    obs_at_rank_one_for_klo = obs_nearest_k.wind_speed.loc[:, "KLO", 1].values
    np.testing.assert_equal(obs_at_rank_one_for_klo_original, obs_at_rank_one_for_klo)


def test_variable_select_rank(raw_obs_dataset):

    k = 5
    rank = 0

    obs_dataset = raw_obs_dataset()[["wind_speed"]]
    stations = stations_df_from_obs_ds(obs_dataset)
    obs_nearest_k = obs_dataset.preproc.euclidean_nearest_k(stations, k)
    obs_best_rank = obs_nearest_k.preproc.select_rank(rank)
    assert list(obs_best_rank.dims) == ["time", "station"]

    # check that missing samples are subsituted by a following ranked neighbor
    nan_idx = np.argwhere(np.isnan(obs_dataset.wind_speed.values))
    non_zero_rank_idx = np.argwhere(obs_best_rank.neighbor_rank.values != 0)
    assert np.all(nan_idx == non_zero_rank_idx)

    # check that the subsitute samples correspond to the original ones
    nan_idx = list(
        zip(
            obs_dataset.time[nan_idx[:, 0]].values,
            obs_dataset.station[nan_idx[:, 1]].values,
        )
    )
    nan_subs = (
        obs_best_rank.stack(sample=["time", "station"]).sel(sample=nan_idx).wind_speed
    )
    nan_subs_idx = list(zip(nan_subs.time.values, nan_subs.neighbor.values))
    nan_sub_original = (
        obs_dataset.stack(sample=["time", "station"])
        .sel(sample=nan_subs_idx)
        .wind_speed
    )
    np.testing.assert_equal(nan_subs.values, nan_sub_original.values)
