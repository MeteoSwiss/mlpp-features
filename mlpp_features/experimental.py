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

    # Compute the distance from the point to the closest point
    return np.linalg.norm(point - closest_point)


def signs_points_to_line(
    points: List[Tuple[float, float]], line_points: List[Tuple[float, float]]
) -> List[int]:
    """For a list of points, find the sign of the distance to a polyline."""
    signs = []
    min_longitude = min([point[1] for point in line_points])
    max_longitude = max([point[1] for point in line_points])
    print(min_longitude, max_longitude)
    for point in points:
        if point[1] < min_longitude:
            segment_latitude = line_points[list(list(zip(*line_points))[1]).index(min_longitude)][0]
            print(segment_latitude, point[0] - segment_latitude)
            sign = np.sign(point[0] - segment_latitude)
        elif point[1] > max_longitude:
            segment_latitude = line_points[list(list(zip(*line_points))[1]).index(max_longitude)][0]
            print(segment_latitude, point[0] - segment_latitude)
            sign = np.sign(point[0] - segment_latitude)
        else:
            for i in range(len(line_points) - 1):
                segment_start = line_points[i]
                segment_end = line_points[i + 1]
                if point[1] >= segment_start[1] and point[1] <= segment_end[1]:
                    slope = (segment_end[0] - segment_start[0]) / (segment_end[1] - segment_start[1])
                    lat_intercept = segment_start[0] - slope * segment_start[1]
                    y_proj_point_to_segment = slope * point[1] + lat_intercept
                    if point[0] >= y_proj_point_to_segment:
                        sign = 1
                    else:
                        sign = -1
        signs.append(sign)
    return signs


def distances_points_to_line(
    points: List[Tuple[float, float]], line_points: List[Tuple[float, float]]
) -> List[float]:
    """For a list of points, find the minimum distance a polyline."""
    min_distances = []
    signs = signs_points_to_line(points, line_points)
    for point in points:
        min_distance = float("inf")
        for i in range(len(line_points) - 1):
            segment_start = line_points[i]
            segment_end = line_points[i + 1]
            distance = distance_point_to_segment(point, segment_start, segment_end)
            if np.abs(distance) < np.abs(min_distance):
                min_distance = distance
        min_distances.append(min_distance)
    return np.array(min_distances)*np.array(signs)
