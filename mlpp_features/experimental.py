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


def distance_point_to_segment_old(
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
    sign = np.sign(point[1] - closest_point[1])
    
    return np.linalg.norm(point - closest_point), sign


def haversine(latp, lonp, latp2, lonp2, **kwargs):
    """──────────────────────────────────────────────────────────────────────────┐
      Haversine formula for calculating distance between two points (latp,
      lonp) and (latp2, lonp2). This function can handle
      2D lat/lon lists, but has been used with flattened data

      Based on:
      https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97


      Inputs:
          latp - latitude of target point

          lonp - longitude of target point

          latp2 - latitude of second point

          lonp2 - longitude of second point

      Outputs:

    └──────────────────────────────────────────────────────────────────────────"""
    kwargs.get("epsilon", 1e-6)

    latp = np.radians(latp)
    lonp = np.radians(lonp)
    latp2 = np.radians(latp2)
    lonp2 = np.radians(lonp2)

    dlon = lonp - lonp2
    dlat = latp - latp2
    a = np.power(np.sin(dlat / 2), 2) + np.cos(latp2) * np.cos(latp) * np.power(
        np.sin(dlon / 2), 2
    )

    # Assert that sqrt(a) is within machine precision of 1
    # assert np.all(np.sqrt(a) <= 1 + epsilon), 'Invalid argument for arcsin'

    # Check if square root of a is a valid argument for arcsin within machine precision
    # If not, set to 1 or -1 depending on sign of a
    a = np.where(np.sqrt(a) <= 1, a, np.sign(a))

    return 2 * 6371 * np.arcsin(np.sqrt(a))


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
    
    return haversine(*point, *closest_point)


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


def get_sign_for_inside_point(point: Tuple[float, float], segment_start: Tuple[float, float], segment_end: Tuple[float, float]) -> int:
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



def signs_points_to_line(
    points: List[Tuple[float, float]], line_points: List[Tuple[float, float]]
) -> List[int]:
    """
    For a list of points, find the sign of the distance to a polyline.
    Inputs:
        - points: list of points (latitude, longitude)
        - line_points: list of points defining the polyline (latitude, longitude)

    Outputs:
        - signs: list of signs (+1 or -1) depending if the point is above (north) or below (south) the polyline
    
    """
    signs = []
    
    # Extract the west and east most points
    longitudes = list(zip(*line_points))[1]
    
    west_most_point = line_points[list(longitudes).index(min(longitudes))]
    east_most_point = line_points[list(longitudes).index(max(longitudes))]

    for point in points:
        if point[1] < min(longitudes):
            sign = get_sign_for_outside_point(point, west_most_point)
        elif point[1] > max(longitudes):
            sign = get_sign_for_outside_point(point, east_most_point)
        else:
            for i in range(len(line_points) - 1):
                segment_start = line_points[i]
                segment_end = line_points[i + 1]
                if point[1] >= segment_start[1] and point[1] <= segment_end[1]:
                    sign = get_sign_for_inside_point(point, segment_start, segment_end)
                    break
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
            if distance < np.abs(min_distance):
                min_distance = distance

        min_distances.append(min_distance)
    return np.array(min_distances) * signs
