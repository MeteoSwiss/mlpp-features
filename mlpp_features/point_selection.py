import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

from scipy.spatial import KDTree

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


LOGGER = logging.getLogger(__name__)


class PointSelector(ABC):
    """Represent a point neighbor selector method."""

    @abstractmethod
    def query(self, coords, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the nearest grid cell to a given point.

        Parameters
        ----------
        points: tuple of lists
            longitude, latitude(, height)

        Return
        ------
        ravel_index: array_like, shape (<=n,)
        mask: array_like, shape (n,)
        """


@dataclass
class EuclideanNearestRegular(PointSelector):
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
        src_crs = self.dataset.attrs["crs"]
        self.x_coords = self.dataset.x.values
        self.y_coords = self.dataset.y.values
        src_proj = CRS(src_crs)
        dst_proj = CRS(self.dst_crs)
        if not (src_proj.to_epsg() == dst_proj.to_epsg()):
            # TODO: should give it more thoughts, but to be safe, let's not allow
            # reprojecting 1D grid coordinated for now...
            raise NotImplementedError("cannot reproject 1D grid coordinates")
        self.transformer = Transformer.from_crs(
            CRS("epsg:4326"), dst_proj, always_xy=True
        )
        if self.grid_res is None:
            x_grid_res = np.abs(np.gradient(self.x_coords)).mean()
            y_grid_res = np.abs(np.gradient(self.y_coords)).mean()
            self.grid_res = np.mean((x_grid_res, y_grid_res))
        if "HSURF" in self.dataset:
            self.hsurf = self.dataset.HSURF.values
        if "FR_LAND" in self.dataset:
            self.fr_land = self.dataset.FR_LAND.values
        del self.dataset

    def query(self, coords, search_radius=1.415, vertical_weight=0):

        longitude, latitude = coords[:2]
        x_coords, y_coords = self.transformer.transform(longitude, latitude)

        # Approximate a safe number of nearest neighbors and the
        # corresponding maximum distance
        k = int(np.ceil(2 * search_radius) ** 2)
        max_distance = search_radius * self.grid_res

        x_grid = np.expand_dims(self.x_coords, axis=1)
        x_points = np.expand_dims(x_coords, axis=0)
        x_diff = np.abs(x_grid - x_points)
        x_index = np.argpartition(x_diff, range(k), axis=0)[:k]

        y_grid = np.expand_dims(self.y_coords, axis=1)
        y_stations = np.expand_dims(y_coords, axis=0)
        y_diff = np.abs(y_grid - y_stations)
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
            height = coords[2]
            hsurf = self.hsurf[y_index, x_index]
            height_diff = np.abs(hsurf - height)
            distance = horizontal_distance + vertical_weight * height_diff
        else:
            distance = horizontal_distance.copy()

        # Query nearest neighbour
        distance[~valid] *= 1e6
        index = (np.argmin(distance, axis=0), np.arange(distance.shape[-1]))
        distance = horizontal_distance[index]
        x_index = x_index[index]
        y_index = y_index[index]

        # Filter out stations that are too distant
        x_index = x_index[distance < max_distance]
        y_index = y_index[distance < max_distance]

        mask = np.array([True if d < max_distance else False for d in distance])

        index = np.ravel_multi_index(
            (y_index, x_index),
            (self.y_coords.size, self.x_coords.size),
        )
        return index, mask


@dataclass
class EuclideanNearestIrregular(PointSelector):
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
        if "HSURF" in self.dataset:
            self.hsurf = self.dataset.HSURF.values
        if "FR_LAND" in self.dataset:
            self.fr_land = self.dataset.FR_LAND.values
        del self.dataset

    def query(self, coords, search_radius=1.415, vertical_weight=0):

        longitude, latitude = coords[:2]
        xy_coords = np.column_stack(self.transformer.transform(longitude, latitude))

        # Approximate a safe number of nearest neighbors and the
        # corresponding maximum distance
        k = int(np.ceil(2 * search_radius) ** 2)
        max_distance = search_radius * self.grid_res

        horizontal_distance, index = self.tree.query(xy_coords, k)
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
            height = coords[2][:, None]
            hsurf = self.hsurf.ravel()[index]
            height_diff = np.abs(hsurf - height)
            distance = horizontal_distance + vertical_weight * height_diff
        else:
            distance = horizontal_distance.copy()

        # Query nearest neighbour
        distance[~valid] *= 1e6
        index_ = (np.arange(distance.shape[0]), np.argmin(distance, axis=1))
        distance = horizontal_distance[index_]

        mask = np.array([True if d < max_distance else False for d in distance])

        # Filter out stations that are too distant
        index = index[index_][distance < max_distance]

        return index, mask

# NOTE: this is a first component for the nowcasting, it works by passing the dwh dataset (dwh.zarr)
# and then query a list of points derived by the dwh dataset itself

@dataclass
class EuclideanNearestSparse(PointSelector):
    """Get Euclidean nearest neighbor in the case of a sparse collection of points."""

    dataset: xr.Dataset = field(repr=False)
    dst_crs: str = "epsg:21781"
    grid_res: Optional[float] = None

    # Derived variables (init=False)
    dst_proj: CRS = field(init=False, repr=False)
    tranformer: Transformer = field(init=False, repr=False)
    coords: np.ndarray = field(init=False, repr=False)
    tree: Optional[KDTree] = field(init=False, repr=False, default=None)
    height: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self):
        latitude = self.dataset["station_lat"].values
        longitude = self.dataset["station_lon"].values
        src_proj = CRS("epsg:4326")
        dst_proj = CRS(self.dst_crs)
        self.transformer = Transformer.from_crs(src_proj, dst_proj, always_xy=True)
        x_coords, y_coords = self.transformer.transform(longitude, latitude)
        self.coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
        self.tree = KDTree(self.coords)

        self.height = self.dataset["station_height"].values

        del self.dataset

    def query(self, coords, k=5, vertical_weight=0):

        longitude, latitude = coords[:2]
        xy_coords = np.column_stack(self.transformer.transform(longitude, latitude))

        horizontal_distance, index = self.tree.query(xy_coords, k)
        horizontal_distance = np.array(horizontal_distance)
        index = np.array(index)

        # Include vertical emphasis
        if vertical_weight > 0:
            height_points = coords[2][:, None]
            height = self.height.ravel()[index]
            height_diff = np.abs(height - height_points)
            distance = horizontal_distance + vertical_weight * height_diff
        else:
            distance = horizontal_distance.copy()

        # Query nearest neighbour
        sorted_distance = np.argsort(distance, axis=1)
        distance = np.take_along_axis(distance, sorted_distance, axis = 1)
        index = np.take_along_axis(index, sorted_distance, axis = 1)

        return distance, index