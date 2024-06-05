import pandas as pd
import pytest
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlpp_features.experimental as exp

def test_sign_distance_to_alpine_ridge():
    """Test distance to alpine ridge with sign"""

    rigde1 = [(46.75, 6.0), (46.75, 11.0)] # horizontal line
    ridge2 = [(45.5, 6.0), (47.0, 11.0)] # diagonal line
    ridge3 = [(47.0, 6.0), (45.5, 11.0)] # diagonal line
    ridge4 = [[45.67975, 6.88306],
              [45.75149, 6.80643],
              [45.88912, 7.07724],
              [45.86909, 7.17029],
              [46.25074, 8.03064],
              [46.47280, 8.38946],
              [46.55972, 8.55968],
              [46.56318, 8.80080],
              [46.61256, 8.96059],
              [46.49712, 9.17104],
              [46.50524, 9.33031],
              [46.39905, 9.69325],
              [46.40885, 10.01963],
              [46.63982, 10.29218],
              [46.83630, 10.50783],
              [46.90567, 11.09742],
    ] # the actual alpine ridge
    
    # generate 400 random points
    rand_lat = np.random.uniform(45.5, 47.0, 20)
    rand_lon = np.random.uniform(6.0, 11.0, 20)
    lat_coords, lon_coords = np.meshgrid(rand_lat, rand_lon)
    rand_points = list(zip(lat_coords.ravel(), lon_coords.ravel()))

    # compute signs for the 4 ridges

    signs1 = np.sign(exp.distances_points_to_line(rand_points, rigde1))
    signs2 = np.sign(exp.distances_points_to_line(rand_points, ridge2))
    signs3 = np.sign(exp.distances_points_to_line(rand_points, ridge3))
    signs4 = np.sign(exp.distances_points_to_line(rand_points, ridge4))

    # create dataframe to easily plot the data

    df1 = pd.DataFrame({"latitude": lat_coords.ravel(), "longitude": lon_coords.ravel(), "sign": signs1})
    df2 = pd.DataFrame({"latitude": lat_coords.ravel(), "longitude": lon_coords.ravel(), "sign": signs2})
    df3 = pd.DataFrame({"latitude": lat_coords.ravel(), "longitude": lon_coords.ravel(), "sign": signs3})
    df4 = pd.DataFrame({"latitude": lat_coords.ravel(), "longitude": lon_coords.ravel(), "sign": signs4})

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    sns.scatterplot(
        ax=axs[0, 0], data=df1, x="longitude", y="latitude", hue="sign"
    )
    axs[0, 0].plot([rigde1[i][1] for i in range(len(rigde1))], [rigde1[i][0] for i in range(len(rigde1))], color="red")

    sns.scatterplot(
        ax=axs[0, 1], data=df2, x="longitude", y="latitude", hue="sign"
    )
    axs[0, 1].plot([ridge2[i][1] for i in range(len(ridge2))], [ridge2[i][0] for i in range(len(ridge2))], color="red")

    sns.scatterplot(
        ax=axs[1, 0], data=df3, x="longitude", y="latitude", hue="sign"
    )
    axs[1, 0].plot([ridge3[i][1] for i in range(len(ridge3))], [ridge3[i][0] for i in range(len(ridge3))], color="red")
    sns.scatterplot(
        ax=axs[1, 1], data=df4, x="longitude", y="latitude", hue="sign"
    )
    axs[1, 1].plot([ridge4[i][1] for i in range(len(ridge4))], [ridge4[i][0] for i in range(len(ridge4))], color="red")

    fig.savefig("./test_sign_distance_to_alpine_ridge.png")

if __name__ == "__main__":
    test_sign_distance_to_alpine_ridge()