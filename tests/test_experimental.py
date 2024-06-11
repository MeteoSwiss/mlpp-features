import pandas as pd
import pytest
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlpp_features.experimental as exp

def test_sign_distance_to_alpine_ridge():
    """Test distance to alpine ridge with sign"""

    ridge1 = [(46, 6.0), (46, 11.0)] # horizontal line
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
    ] # the "actual" alpine ridge
    
    # generate 900 random points
    rand_lat = np.random.uniform(45, 47.5, 30)
    rand_lon = np.random.uniform(5.5, 11.5, 30)
    lat_coords, lon_coords = np.meshgrid(rand_lat, rand_lon)
    rand_points = list(zip(lat_coords.ravel(), lon_coords.ravel()))

    # compute signs for the 4 ridges

    signs1 = np.sign(exp.distances_points_to_line(rand_points, ridge1))
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
    axs[0, 0].plot([ridge1[i][1] for i in range(len(ridge1))], [ridge1[i][0] for i in range(len(ridge1))], color="red")

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

    fig.suptitle("Sign of distance to alpine ridge for different configuration of the Alpine ridge (red line)")
    fig.tight_layout()
    fig.savefig("./test_sign_distance_to_alpine_ridge.png", bbox_inches="tight")


def test_distance_to_alpine_ridge():
    """Test distance to alpine ridge"""

    ridge1 = [(46, 6.0), (46, 11.0)] # horizontal line
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
    ] # the "actual" alpine ridge
    
    # generate 10000 random points
    rand_lat = np.random.uniform(45.5, 47.0, 100)
    rand_lon = np.random.uniform(6.0, 11.0, 100)
    all_ridges_lat, all_ridges_lon = [], []
    for ridge in [ridge1, ridge2, ridge3, ridge4]:
        all_ridges_lat.extend([r[0] for r in ridge])
        all_ridges_lon.extend([r[1] for r in ridge])

    latitudes, longitudes = np.unique(np.concatenate((rand_lat, all_ridges_lat))), np.unique(np.concatenate((rand_lon, all_ridges_lon)))
    lat_coords, lon_coords = np.meshgrid(latitudes, longitudes)

    rand_points = list(zip(lat_coords.ravel(), lon_coords.ravel()))

    # compute distances for the 4 ridges

    dist1 = exp.distances_points_to_line(rand_points, ridge1)
    dist2 = exp.distances_points_to_line(rand_points, ridge2)
    dist3 = exp.distances_points_to_line(rand_points, ridge3)
    dist4 = exp.distances_points_to_line(rand_points, ridge4)

    # create dataframe to easily plot the data

    df1 = pd.DataFrame({"latitude": lat_coords.ravel(), "longitude": lon_coords.ravel(), "distance": dist1})
    df2 = pd.DataFrame({"latitude": lat_coords.ravel(), "longitude": lon_coords.ravel(), "distance": dist2})
    df3 = pd.DataFrame({"latitude": lat_coords.ravel(), "longitude": lon_coords.ravel(), "distance": dist3})
    df4 = pd.DataFrame({"latitude": lat_coords.ravel(), "longitude": lon_coords.ravel(), "distance": dist4})

    fig, axs = plt.subplots(2, 2, figsize=(13, 13))
    
    
    im1 = axs[0, 0].imshow(df1.pivot(index="latitude", columns="longitude", values="distance"),
                           cmap="twilight", origin="lower", vmin=-df1["distance"].abs().max(), vmax=df1["distance"].abs().max())
    fig.colorbar(im1, ax=axs[0, 0], shrink=0.8)
    axs[0, 0].set_title("Horizontal ridge")
    axs[0, 0].set_xticks(range(len(longitudes))[::10], labels=[round(lon, 2) for lon in longitudes][::10])
    axs[0, 0].set_yticks(range(len(latitudes))[::10], labels=[round(lat, 2) for lat in latitudes][::10])
    axs[0, 0].set_xlabel("Longitude")
    axs[0, 0].set_ylabel("Latitude")

    im2 = axs[0, 1].imshow(df2.pivot(index="latitude", columns="longitude", values="distance"),
                           cmap="twilight", origin="lower", vmin=-df1["distance"].abs().max(), vmax=df1["distance"].abs().max())
    fig.colorbar(im2, ax=axs[0, 1], shrink=0.8)
    axs[0, 1].set_title("Diagonal ridge")
    axs[0, 1].set_xticks(range(len(longitudes))[::10], labels=[round(lon, 2) for lon in longitudes][::10])
    axs[0, 1].set_yticks(range(len(latitudes))[::10], labels=[round(lat, 2) for lat in latitudes][::10])
    axs[0, 1].set_xlabel("Longitude")
    axs[0, 1].set_ylabel("Latitude")

    im3 = axs[1, 0].imshow(df3.pivot(index="latitude", columns="longitude", values="distance"),
                           cmap="twilight", origin="lower", vmin=-df1["distance"].abs().max(), vmax=df1["distance"].abs().max())
    fig.colorbar(im3, ax=axs[1, 0], shrink=0.8)
    axs[1, 0].set_title("Diagonal ridge")
    axs[1, 0].set_xticks(range(len(longitudes))[::10], labels=[round(lon, 2) for lon in longitudes][::10])
    axs[1, 0].set_yticks(range(len(latitudes))[::10], labels=[round(lat, 2) for lat in latitudes][::10])
    axs[1, 0].set_xlabel("Longitude")
    axs[1, 0].set_ylabel("Latitude")

    im4 = axs[1, 1].imshow(df4.pivot(index="latitude", columns="longitude", values="distance"),
                           cmap="twilight", origin="lower", vmin=-df1["distance"].abs().max(), vmax=df1["distance"].abs().max())
    fig.colorbar(im4, ax=axs[1, 1], shrink=0.8)
    axs[1, 1].set_title("Alpine ridge")
    axs[1, 1].set_xticks(range(len(longitudes))[::10], labels=[round(lon, 2) for lon in longitudes][::10])
    axs[1, 1].set_yticks(range(len(latitudes))[::10], labels=[round(lat, 2) for lat in latitudes][::10])
    axs[1, 1].set_xlabel("Longitude")
    axs[1, 1].set_ylabel("Latitude")

    fig.suptitle("Distance to alpine ridge for different configuration of the Alpine ridge")
    fig.tight_layout()
    fig.savefig("./test_distance_to_alpine_ridge.png", bbox_inches="tight")


if __name__ == "__main__":
    test_sign_distance_to_alpine_ridge()
    test_distance_to_alpine_ridge()