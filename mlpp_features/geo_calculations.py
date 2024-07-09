"""This module provides functions for geospatial calculations."""

from typing import List, Tuple

import numpy as np
from pyproj import CRS, Transformer


RADIUS_EARTH_KM = 6371.0


def reproject_points(
    latlon_wgs84: List[Tuple[float, float]], dst_epsg: str
) -> List[Tuple[float, float]]:
    transformer = Transformer.from_crs(CRS("epsg:4326"), CRS(dst_epsg), always_xy=True)
    lon_src = [p[1] for p in latlon_wgs84]
    lat_src = [p[0] for p in latlon_wgs84]
    x_dst, y_dst = transformer.transform(lon_src, lat_src)
    return [(x, y) for x, y in zip(x_dst, y_dst)]


def calculate_haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Haversine formula for calculating distance between two points (latp, lonp) and
    (latp2, lonp2). This function can handle 2D lat/lon lists, but has been used with
    flattened data

    Based on:
    https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97


    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: Distance between the two points in kilometers.

    """
    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Compute differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Apply Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) + np.sin(dlon / 2) ** 2

    # Ensure the argument for arcsin is within the valid range [-1, 1]
    a = np.clip(a, -1, 1)

    # Check if square root of a is a valid argument for arcsin within machine precision
    # If not, set to 1 or -1 depending on sign of a
    a = np.where(np.sqrt(a) <= 1, a, np.sign(a))

    # Calculate distance
    distance = 2 * RADIUS_EARTH_KM * np.arcsin(np.sqrt(a))

    return distance


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
    return calculate_haversine_distance(*point, *closest_point)


def get_sign_for_outside_point(point: Tuple[float, float], extreme_line_point) -> int:
    """
    For a point outside the line, find the sign of the distance to the line.
    Inputs:
        - point: point outside (west or east to the west-most/east-most point) the line (latitude, longitude)
        - extreme_line_point: extreme point of the line (latitude, longitude)

    Outputs:
        - sign: sign (+1 or -1) depending if the point is above (north) or below (south) the extreme point

    """

    s = np.sign(point[0] - extreme_line_point[0])
    s = s if s != 0 else 1
    return s


def get_sign_for_inside_point(
    point: Tuple[float, float],
    segment_start: Tuple[float, float],
    segment_end: Tuple[float, float],
) -> int:
    """
    For a point inside a segment of the line (i.e., lon_segment_start <= lon_point <= long_segment_end),
    find the sign of the distance to the line.
    Inputs:
        - point: point inside the line (latitude, longitude)
        - segment_start: start point of the segment (latitude, longitude)
        - segment_end: end point of the segment (latitude, longitude)

    Outputs:
        - sign: sign (+1 or -1) depending if the point is above (north) or below (south) the segment

    """
    slope = (segment_end[0] - segment_start[0]) / (segment_end[1] - segment_start[1])
    lat_intercept = segment_start[0] - slope * segment_start[1]
    y_proj_point_to_segment = slope * point[1] + lat_intercept
    if point[0] >= y_proj_point_to_segment:
        sign = 1
    else:
        sign = -1
    return sign


def sign_point_to_segment(
    point: Tuple[float, float],
    segment_start: Tuple[float, float],
    segment_end: Tuple[float, float],
) -> int:
    """
    For a given points, find the sign of the distance to a segment.
    Returns a value only if the point is "inside" the segment.
    Inputs:
        - point: (latitude, longitude) of the considered station
        - segment_start: start point of the segment (latitude, longitude)
        - segment_end: end point of the segment (latitude, longitude)

    Outputs:
        - sign: +1 or -1 depending if the point is above (north) or below (south) the segment
                None if the point is outside the segment

    """

    if point[1] >= min(segment_start[1], segment_end[1]) and point[1] <= max(
        segment_start[1], segment_end[1]
    ):
        sign = get_sign_for_inside_point(point, segment_start, segment_end)
    else:
        sign = None

    return sign


def line_extreme_points(
    line_points: List[Tuple[float, float]]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Find the west-most and east-most points of a polyline.
    Inputs:
        - line_points: list of points defining the polyline (latitude, longitude)

    Outputs:
        - west_most_point: west-most point of the polyline (latitude, longitude)
        - east_most_point: east-most point of the polyline (latitude, longitude)

    """
    longitudes = list(zip(*line_points))[1]
    west_most_point = line_points[list(longitudes).index(min(longitudes))]
    east_most_point = line_points[list(longitudes).index(max(longitudes))]
    return west_most_point, east_most_point


def distances_points_to_line(
    points: List[Tuple[float, float]], line_points: List[Tuple[float, float]]
) -> List[float]:
    """For a list of points, find the minimum distance a polyline."""
    min_distances = []
    west_most_point, east_most_point = line_extreme_points(line_points)
    for point in points:
        min_distance = float("inf")
        tmp_signs = []
        for i in range(len(line_points) - 1):
            segment_start = line_points[i]
            segment_end = line_points[i + 1]
            distance = distance_point_to_segment(point, segment_start, segment_end)
            if point[1] < west_most_point[1]:
                sign = get_sign_for_outside_point(point, west_most_point)
            elif point[1] > east_most_point[1]:
                sign = get_sign_for_outside_point(point, east_most_point)
            else:
                sign = sign_point_to_segment(point, segment_start, segment_end)
            if sign is not None:
                tmp_signs.append(sign)

            if distance < min_distance:
                min_distance = distance
        sign = [k if np.allclose(tmp_signs, k) else -1 for k in set(tmp_signs)][0]
        min_distances.append(min_distance * sign)
    return np.array(min_distances)
