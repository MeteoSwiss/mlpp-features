"""This module provides functions for geospatial calculations."""

from typing import List, Tuple

import numpy as np
from pyproj import CRS, Transformer


RADIUS_EARTH_KM = 6371.0


def reproject_points(
    latlon_wgs84: List[Tuple[float, float]], dst_epsg: str
) -> List[Tuple[float, float]]:
    """
    Reproject a list of latitude/longitude points from WGS84 to a specified EPSG coordinate system.

    Args:
        latlon_wgs84 (List[Tuple[float, float]]): List of points in WGS84 format (latitude, longitude).
        dst_epsg (str): EPSG code of the target coordinate system.

    Returns:
        List[Tuple[float, float]]: List of reprojected points in the target coordinate system.
    """
    transformer = Transformer.from_crs(CRS("epsg:4326"), CRS(dst_epsg), always_xy=True)
    lon_src = [p[1] for p in latlon_wgs84]
    lat_src = [p[0] for p in latlon_wgs84]
    x_dst, y_dst = transformer.transform(lon_src, lat_src)
    return [(x, y) for x, y in zip(x_dst, y_dst)]


def calculate_haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great-circle distance between two points on the Earth's surface using the Haversine formula.

    This function can handle 2D lat/lon lists, but has been used with flattened data.

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
    dlon = lon1 - lon2
    dlat = lat1 - lat2

    # Apply Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    # Ensure the argument for arcsin is within the valid range [-1, 1]
    a = np.clip(a, -1, 1)

    # Calculate distance
    distance = 2 * RADIUS_EARTH_KM * np.arcsin(np.sqrt(a))

    return distance


def distance_point_to_segment(
    point: Tuple[float, float],
    segment_start: Tuple[float, float],
    segment_end: Tuple[float, float],
) -> float:
    """
    Calculate the minimum distance from a point to a line segment.

    Args:
        point (Tuple[float, float]): The point (latitude, longitude).
        segment_start (Tuple[float, float]): The start point of the segment (latitude, longitude).
        segment_end (Tuple[float, float]): The end point of the segment (latitude, longitude).

    Returns:
        float: The minimum distance from the point to the line segment.
    """
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
    For a point outside the line, determine the sign of the distance to the line.

    Args:
        point (Tuple[float, float]): Point outside the line (latitude, longitude).
        extreme_line_point (Tuple[float, float]): Extreme point of the line (latitude, longitude).

    Returns:
        int: +1 if the point is north of the line, -1 if the point is south of the line.
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
    For a point inside the line, determine the sign of the distance to the line.

    Inside the line means that lon_segment_start <= lon_point <= long_segment_end.

    Args:
        point (Tuple[float, float]): Point inside the segment (latitude, longitude).
        segment_start (Tuple[float, float]): Start point of the segment (latitude, longitude).
        segment_end (Tuple[float, float]): End point of the segment (latitude, longitude).

    Returns:
        int: +1 if the point is north of the segment, -1 if the point is south of the segment.
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
    Determine the sign of the distance from a point to a segment if the point is within the segment's longitude bounds.

    Args:
        point (Tuple[float, float]): The point (latitude, longitude).
        segment_start (Tuple[float, float]): Start point of the segment (latitude, longitude).
        segment_end (Tuple[float, float]): End point of the segment (latitude, longitude).

    Returns:
        int: +1 if the point is north of the segment, -1 if the point is south of the segment, None if outside the segment.
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

    Args:
        line_points (List[Tuple[float, float]]): List of points defining the polyline (latitude, longitude).

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float]]: West-most and east-most points of the polyline.
    """
    longitudes = list(zip(*line_points))[1]
    west_most_point = line_points[list(longitudes).index(min(longitudes))]
    east_most_point = line_points[list(longitudes).index(max(longitudes))]
    return west_most_point, east_most_point


def distances_points_to_line(
    points: List[Tuple[float, float]], line_points: List[Tuple[float, float]]
) -> List[float]:
    """
    Calculate the minimum distance from each point in a list to a polyline.

    Args:
        points (List[Tuple[float, float]]): List of points (latitude, longitude).
        line_points (List[Tuple[float, float]]): List of points defining the polyline (latitude, longitude).

    Returns:
        List[float]: List of minimum distances from each point to the polyline.
    """
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
