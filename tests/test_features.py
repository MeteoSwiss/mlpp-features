import numpy as np

import mlpp_features  # type: ignore


def stations_df_from_obs_ds(obs_dataset):
    stations = obs_dataset[
        ["station_name", "station_lon", "station_lat", "station_height"]
    ].to_pandas()
    stations = stations.rename(
        columns={
            "station_name": "name",
            "station_lon": "longitude",
            "station_lat": "latitude",
            "station_height": "elevation",
        }
    )
    return stations.reset_index().set_index("name").drop(columns="variable")


def test_variable_euclidean_nearest_k(raw_obs_dataset):

    k = 5
    obs_dataset = raw_obs_dataset()[["measurement"]].sel(variable="wind_speed")
    stations = stations_df_from_obs_ds(obs_dataset)
    obs_nearest_k = obs_dataset.preproc.euclidean_nearest_k(stations, k)

    good_dims = ["time", "station", "neighbor_rank"]
    assert list(obs_nearest_k.dims) == good_dims

    obs_at_aaa_original = obs_dataset.measurement.loc[:, 0].values
    obs_at_aaa_rank_zero = obs_nearest_k.measurement.loc[:, "AAA", 0].values
    np.testing.assert_equal(obs_at_aaa_original, obs_at_aaa_rank_zero)

    obs_at_rank_one_for_aaa_original = (
        obs_dataset.swap_dims({"station_id": "station_name"})
        .measurement.loc[:, obs_nearest_k.neighbor_name.loc["AAA", 1]]
        .values
    )

    obs_at_rank_one_for_aaa = obs_nearest_k.measurement.loc[:, "AAA", 1].values
    np.testing.assert_equal(obs_at_rank_one_for_aaa_original, obs_at_rank_one_for_aaa)


def test_variable_select_rank(raw_obs_dataset):

    k = 5
    rank = 0

    obs_dataset = raw_obs_dataset()[["measurement"]].sel(variable="wind_speed")
    stations = stations_df_from_obs_ds(obs_dataset)
    obs_nearest_k = obs_dataset.preproc.euclidean_nearest_k(stations, k)
    obs_best_rank = obs_nearest_k.preproc.select_rank(rank)

    good_dims = ["time", "station"]
    assert list(obs_best_rank.dims) == good_dims

    # check that missing samples are subsituted by a following ranked neighbour
    nan_idx = np.argwhere(np.isnan(obs_dataset.measurement.values))
    non_zero_rank_idx = np.argwhere(obs_best_rank.neighbor_rank.values != 0)
    assert np.all(nan_idx == non_zero_rank_idx)

    # check that the subsitute samples correspond to the original ones
    nan_idx = list(
        zip(
            obs_dataset.time[nan_idx[:, 0]].values,
            obs_dataset.station_name[nan_idx[:, 1]].values,
        )
    )
    nan_subs = (
        obs_best_rank.stack(sample=["time", "station"]).sel(sample=nan_idx).measurement
    )
    nan_subs_idx = list(zip(nan_subs.time.values, nan_subs.neighbor_name.values))
    nan_sub_original = (
        obs_dataset.swap_dims({"station_id": "station_name"})
        .stack(sample=["time", "station_name"])
        .sel(sample=nan_subs_idx)
        .measurement
    )
    np.testing.assert_equal(nan_subs.values, nan_sub_original.values)
