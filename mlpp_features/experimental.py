"""This module implements experimental features"""

from typing import List, Tuple

import numpy as np
from pyproj import CRS, Transformer


def reproject_points(
    latlon_wgs84: List[Tuple[float, float]], dst_epsg: str
) -> List[Tuple[float, float]]:
    transformer = Transformer.from_crs(CRS("epsg:4326"), CRS(dst_epsg), always_xy=True)
    lon_src = [p[1] for p in latlon_wgs84]
    lat_src = [p[0] for p in latlon_wgs84]
    x_dst, y_dst = transformer.transform(lon_src, lat_src)
    return [(x, y) for x, y in zip(x_dst, y_dst)]


def distance_point_to_segment(
    point: Tuple[float, float],
    segment_start: Tuple[float, float],
    segment_end: Tuple[float, float],
) -> float:
    """Calculate the distance from a point to a line segment."""
    point = np.array(point)
    segment_start = np.array(segment_start)
    segment_end = np.array(segment_end)

    # Vector from start to point and start to end
    start_to_point = point - segment_start
    start_to_end = segment_end - segment_start

    # Project start_to_point onto start_to_end
    projection = np.dot(start_to_point, start_to_end) / np.dot(
        start_to_end, start_to_end
    )

    # Check if projection is in the segment
    if projection < 0:
        closest_point = segment_start
    elif projection > 1:
        closest_point = segment_end
    else:
        closest_point = segment_start + projection * start_to_end
    
    # compute north/south sign wrt to segment
    # By default, we assign a positive sign for points on the line
    if point[0] - closest_point[0] > 0:
        sign = -1
    elif point[0] - closest_point[0] < 0:
        sign = 1
    else:
        if point[1] - closest_point[1] > 0:
            sign = -1
        else:
            sign = 1

    # Compute the distance from the point to the closest point
    return sign * np.linalg.norm(point - closest_point)


def distances_points_to_line(
    points: List[Tuple[float, float]], line_points: List[Tuple[float, float]]
) -> List[float]:
    """For a list of points, find the minimum distance a polyline."""
    min_distances = []
    for point in points:
        min_distance = float("inf")
        for i in range(len(line_points) - 1):
            segment_start = line_points[i]
            segment_end = line_points[i + 1]
            distance = distance_point_to_segment(point, segment_start, segment_end)
            if np.abs(distance) < np.abs(min_distance):
                min_distance = distance
        min_distances.append(min_distance)
    return min_distances
