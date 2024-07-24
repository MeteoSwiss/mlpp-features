import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from scipy.spatial import KDTree

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer


LOGGER = logging.getLogger(__name__)


class StationSelector(ABC):
    """Represent a station neighbor selector method."""

    @abstractmethod
    def query(self, stations: pd.DataFrame, **kwargs) -> xr.DataArray:
        """
        Get the indices to the nearest grid cells to a set of stations.

        Parameters
        ----------
        stations: pd.DataFrame
            A collection of stations. The dataframe must include
            the  columns 'longitude', 'latitude', and (optionally)
            'height_masl'.

        Return
        ------
        ravel_index: xr.DataArray
        """


@dataclass
class EuclideanNearestRegular(StationSelector):
    """Get Euclidean nearest neighbor in the case of a regular input grid."""

    dataset: xr.Dataset = field(repr=False)
    dst_crs: str = "epsg:21781"
    grid_res: Optional[float] = None

    # Derived variables (init=False)
    src_proj: CRS = field(init=False, repr=False)
    dst_proj: CRS = field(init=False, repr=False)
    tranformer: Transformer = field(init=False, repr=False)
    x_coords: np.ndarray = field(init=False, repr=False)
    y_coords: np.ndarray = field(init=False, repr=False)
    hsurf: np.ndarray = field(init=False, repr=False, default=None)
    fr_land: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self):
        grid_mapping_attrs = eval(self.dataset.attrs["grid_mapping"])
        if "epsg_code" not in grid_mapping_attrs:
            raise ValueError("missing 'epsg_code' in grid_mapping attribute")
        src_proj = CRS.from_string(grid_mapping_attrs["epsg_code"])
        dst_proj = CRS(self.dst_crs)
        if not (src_proj.to_epsg() == dst_proj.to_epsg()):
            # TODO: should give it more thoughts, but to be safe, let's not allow
            # reprojecting 1D grid coordinated for now...
            raise NotImplementedError("cannot reproject 1D grid coordinates")
        self.transformer = Transformer.from_crs(
            CRS("epsg:4326"), dst_proj, always_xy=True
        )
        self.x_coords = self.dataset.x.values
        self.y_coords = self.dataset.y.values
        if self.grid_res is None:
            x_grid_res = np.abs(np.gradient(self.x_coords)).mean()
            y_grid_res = np.abs(np.gradient(self.y_coords)).mean()
            self.grid_res = np.mean((x_grid_res, y_grid_res))
        if "surface_altitude" in self.dataset:
            self.hsurf = self.dataset.surface_altitude.values
        if "land_area_franction" in self.dataset:
            self.fr_land = self.dataset.land_area_franction.values
        del self.dataset

    def query(self, stations, search_radius=1.415, vertical_weight=0):

        sta_lon = stations["longitude"]
        sta_lat = stations["latitude"]
        sta_x, sta_y = self.transformer.transform(sta_lon, sta_lat)

        # Approximate a safe number of nearest neighbors and the
        # corresponding maximum distance
        k = int(np.ceil(2 * search_radius) ** 2)
        max_distance = search_radius * self.grid_res

        grid_x = np.expand_dims(self.x_coords, axis=1)
        sta_x = np.expand_dims(sta_x, axis=0)
        x_diff = np.abs(grid_x - sta_x)
        x_index = np.argpartition(x_diff, range(k), axis=0)[:k]

        grid_y = np.expand_dims(self.y_coords, axis=1)
        sta_y = np.expand_dims(sta_y, axis=0)
        y_diff = np.abs(grid_y - sta_y)
        y_index = np.argpartition(y_diff, range(k), axis=0)[:k]

        # Also query distance, as to discard values too far away from any
        # grid point
        x_dist = np.take_along_axis(x_diff, x_index, axis=0)
        y_dist = np.take_along_axis(y_diff, y_index, axis=0)
        horizontal_distance = np.linalg.norm(np.array([y_dist, x_dist]), axis=0)

        # Finally query the fraction of land to discard water pixels
        if self.fr_land is not None:
            fr_land = self.fr_land[y_index, x_index]
        else:
            fr_land = np.ones_like(horizontal_distance)

        # Find valid grid points
        valid = np.logical_and(horizontal_distance < max_distance, fr_land > 0.5)

        # Include vertical emphasis
        if vertical_weight > 0:
            sta_height_masl = stations["height_masl"]
            hsurf = self.hsurf[y_index, x_index]
            height_diff = np.abs(hsurf - sta_height_masl)
            distance = horizontal_distance + vertical_weight * height_diff
        else:
            distance = horizontal_distance.copy()

        # Query nearest neighbor
        distance[~valid] *= 1e6
        ind_nearest = (np.argmin(distance, axis=0), np.arange(distance.shape[-1]))
        distance = horizontal_distance[ind_nearest]
        x_index = x_index[ind_nearest]
        y_index = y_index[ind_nearest]

        ravel_index = np.ravel_multi_index(
            (y_index, x_index),
            (self.y_coords.size, self.x_coords.size),
        )
        return xr.DataArray(
            ravel_index,
            dims="station",
            coords={
                "station": stations.index.values,
                "distance": ("station", distance),
                "valid": ("station", distance < max_distance),
            },
        ).astype(int)


@dataclass
class EuclideanNearestIrregular(StationSelector):
    """Get Euclidean nearest neighbor in the case of an irregular input grid."""

    dataset: xr.Dataset = field(repr=False)
    dst_crs: str = "epsg:21781"
    grid_res: Optional[float] = None

    # Derived variables (init=False)
    dst_proj: CRS = field(init=False, repr=False)
    tranformer: Transformer = field(init=False, repr=False)
    coords: np.ndarray = field(init=False, repr=False)
    tree: Optional[KDTree] = field(init=False, repr=False, default=None)
    hsurf: np.ndarray = field(init=False, repr=False, default=None)
    fr_land: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self):
        latitude = self.dataset.latitude.values
        longitude = self.dataset.longitude.values
        src_proj = CRS("epsg:4326")
        dst_proj = CRS(self.dst_crs)
        self.transformer = Transformer.from_crs(src_proj, dst_proj, always_xy=True)
        x_coords, y_coords = self.transformer.transform(longitude, latitude)
        self.coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
        self.tree = KDTree(self.coords)
        if self.grid_res is None:
            x_grid_res = np.abs(np.gradient(x_coords[0, :])).mean()
            y_grid_res = np.abs(np.gradient(y_coords[:, 0])).mean()
            self.grid_res = np.mean((x_grid_res, y_grid_res))
        if "surface_altitude" in self.dataset:
            self.hsurf = self.dataset.surface_altitude.values
        if "land_area_franction" in self.dataset:
            self.fr_land = self.dataset.land_area_franction.values
        del self.dataset

    def query(self, stations, search_radius=1.415, vertical_weight=0):

        sta_lon = stations["longitude"]
        sta_lat = stations["latitude"]
        sta_coords = np.column_stack(self.transformer.transform(sta_lon, sta_lat))

        # Approximate a safe number of nearest neighbors and the
        # corresponding maximum distance
        k = int(np.ceil(2 * search_radius) ** 2)
        max_distance = search_radius * self.grid_res

        horizontal_distance, index = self.tree.query(sta_coords, k)
        horizontal_distance = np.array(horizontal_distance)
        index = np.array(index)

        # Query the fraction of land to discard water pixels
        if self.fr_land is not None:
            fr_land = self.fr_land.ravel()[index]
        else:
            fr_land = np.ones_like(horizontal_distance)

        # Find valid grid points
        valid = np.logical_and(horizontal_distance < max_distance, fr_land > 0.5)

        # Include vertical emphasis
        if vertical_weight > 0:
            sta_height_masl = np.expand_dims(stations["height_masl"], axis=1)
            hsurf = self.hsurf.ravel()[index]
            height_diff = np.abs(hsurf - sta_height_masl)
            distance = horizontal_distance + vertical_weight * height_diff
        else:
            distance = horizontal_distance.copy()

        # Query nearest neighbor
        distance[~valid] *= 1e6
        index_nearest = (np.arange(distance.shape[0]), np.argmin(distance, axis=1))
        distance = horizontal_distance[index_nearest]

        return xr.DataArray(
            index[index_nearest],
            dims="station",
            coords={
                "station": stations.index.values,
                "distance": ("station", distance),
                "valid": ("station", distance < max_distance),
            },
        ).astype(int)


@dataclass
class EuclideanNearestSparse(StationSelector):
    """Get Euclidean nearest neighbor in the case of a sparse collection of stations."""

    dataset: xr.Dataset = field(repr=False)
    dst_crs: str = "epsg:21781"
    grid_res: Optional[float] = None

    # Derived variables (init=False)
    dst_proj: CRS = field(init=False, repr=False)
    tranformer: Transformer = field(init=False, repr=False)
    coords: np.ndarray = field(init=False, repr=False)
    tree: Optional[KDTree] = field(init=False, repr=False, default=None)
    height_masl: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self):
        latitude = self.dataset["latitude"].values
        longitude = self.dataset["longitude"].values
        src_proj = CRS("epsg:4326")
        dst_proj = CRS(self.dst_crs)
        self.transformer = Transformer.from_crs(src_proj, dst_proj, always_xy=True)
        x_coords, y_coords = self.transformer.transform(longitude, latitude)
        self.coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
        self.tree = KDTree(self.coords)

        self.height_masl = self.dataset["height_masl"].values

        del self.dataset

    def query(self, stations, k=5, vertical_weight=0):

        sta_lon = stations["longitude"]
        sta_lat = stations["latitude"]
        sta_coords = np.column_stack(self.transformer.transform(sta_lon, sta_lat))

        horizontal_distance, index = self.tree.query(sta_coords, k)
        horizontal_distance = np.array(horizontal_distance)
        index = np.array(index)

        # Include vertical emphasis
        if vertical_weight > 0:
            height_stations = self.height_masl[:, None]
            height = self.height_masl.ravel()[index]
            height_diff = np.abs(height - height_stations)
            distance = horizontal_distance + vertical_weight * height_diff
        else:
            distance = horizontal_distance.copy()

        # Query nearest neighbors
        sorted_distance = np.argsort(distance, axis=1)
        distance = np.take_along_axis(distance, sorted_distance, axis=1)
        index = np.take_along_axis(index, sorted_distance, axis=1)

        return xr.DataArray(
            index,
            dims=("station", "neighbor_rank"),
            coords={
                "station": stations.index.values,
                "neighbor_rank": range(k),
                "distance": (("station", "neighbor_rank"), distance),
            },
        ).astype(int)
