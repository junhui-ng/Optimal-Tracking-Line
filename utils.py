import cv2
import numpy as np
import random
from typing import List, Tuple


def random_points_in_polygon(polygon: np.ndarray, k: int=1) -> List[Tuple[int, int]]:
    """
    Sample points within polygon, using a rejection sampling approach.
    
    Args:
        polygon (np.ndarray) : Polygon, array with shape N x 2, where 4 is the number of vertices.
        k (int) : Number of points to be sampled.

    Return:
        List[Tuple[int, int]] : A list of randomly sampled (x, y) points, length of list = k.
    """
    points: List[Tuple[int, int]] = list()

    # coordinates of the minimum bounding rectangle (MBR)
    min_x: int = np.min(polygon[:, 0])
    min_y: int = np.min(polygon[:, 1])
    max_x: int = np.max(polygon[:, 0])
    max_y: int = np.max(polygon[:, 1])

    while len(points) < k:
        # randomly generate point within the MBR
        sampled_point = (random.randint(min_x, max_x), random.randint(min_y, max_y))
        # check if the point is within the polygon
        if cv2.pointPolygonTest(polygon, sampled_point, measureDist=False) > 0:
            points.append(sampled_point)

    return points


def relative_position_in_radians(anchor: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the relative position (in radians) of target points with respect to anchor points.

    Args:
        anchor (numpy.ndarray) : Array of shape (n, 2) containing the xy coordinates of points in the reference/anchor set.
        target (numpy.ndarray) : Array of shape (m, 2) containing the xy coordinates of points in the target set
            whose position we want to describe relative to the anchor.
    
    Return:
    float : Angle in radians from anchor centroid to target centroid
    """
    # get centroids of the two clusters
    anchor_centroid = np.mean(anchor, axis=0)
    target_centroid = np.mean(target, axis=0)
    
    # calculate displacement vector (from anchor to target)
    displacement_vector = target_centroid - anchor_centroid
    
    # calculate angle
    angle_radians = np.arctan2(displacement_vector[1], displacement_vector[0])
    
    return angle_radians
