import json
import math
import numpy as np
import random


def arrange_points_in_square(n):
    """
    Arrange n points in a uniform square grid.

    Args:
        n (int): Number of points to arrange.

    Returns:
        list of tuple: 2D coordinates evenly placed in a square pattern.
    """
    side_length = math.ceil(math.sqrt(n))
    rows = side_length
    cols = side_length - 1 if side_length * (side_length - 1) >= n else side_length

    points = []
    for row in range(rows):
        for col in range(cols):
            points.append(((col - cols / 2) * 2, (row - rows / 2) * 2))
    return points


def generate_points_3d(n, region_size=60, max_attempts=1000):
    """
    Generate n 3D points randomly within a cube, with minimum spacing constraints.

    Args:
        n (int): Number of points to generate.
        region_size (float): Cube edge length.
        max_attempts (int): Attempts per point before giving up.

    Returns:
        list of tuple: List of 3D coordinates.
    """
    points = [(0, 0, 0)]  # Start with origin
    random.seed(2020)

    while len(points) < n:
        for _ in range(max_attempts):
            candidate = tuple(
                random.uniform(-region_size / 2, region_size / 2) for _ in range(3)
            )
            distances = [math.dist(p, candidate) for p in points]
            min_distance = min(distances)
            if all(d >= 2 for d in distances) and 3 <= min_distance <= 8:
                points.append(candidate)
                break
        else:
            raise RuntimeError(f"Failed to generate point {len(points)+1} within {max_attempts} attempts")
    return points


def generate_points(n, region_size=50, max_attempts=1000):
    """
    Generate n 2D points with uniform spacing within a bounded square.

    Args:
        n (int): Number of points to generate.
        region_size (float): Square edge length.
        max_attempts (int): Attempts per point before giving up.

    Returns:
        list of tuple: 2D coordinates.
    """
    points = [(0, 0)]
    random.seed(100)

    while len(points) < n:
        for _ in range(max_attempts):
            candidate = (
                random.uniform(-region_size / 2, region_size / 2),
                random.uniform(-region_size / 2, region_size / 2)
            )
            distances = [math.dist(p, candidate) for p in points]
            min_distance = min(distances)
            if all(d >= 2 for d in distances) and 2 <= min_distance <= 8:
                points.append(candidate)
                break
        else:
            raise RuntimeError(f"Failed to generate point {len(points)+1} within {max_attempts} attempts")
    return points


def gen_settings(center, num):
    """
    Generate 2D UAV initialization positions relative to a center.

    Args:
        center (np.ndarray): 2D center position [x, y].
        num (int): Number of UAVs.

    Returns:
        np.ndarray: Array of shape (num, 3), with z=0.
    """
    points = generate_points(num)[:num] + np.array(center)
    return np.array([[x, y, 0] for x, y in points])


def gen_random_2d(num, center, length):
    """
    Generate random 2D points within a square of given length.

    Args:
        num (int): Number of points.
        center (tuple): 2D center (x, y).
        length (float): Side length of square region.

    Returns:
        np.ndarray: Array of (x, y, 0) coordinates.
    """
    half_length = length / 2
    x = np.random.uniform(center[0] - half_length, center[0] + half_length, num)
    y = np.random.uniform(center[1] - half_length, center[1] + half_length, num)
    return np.column_stack((x, y, np.zeros(num)))


def gen_random_3d(num, center, length):
    """
    Generate random 3D points within a cube of given length.

    Args:
        num (int): Number of points.
        center (tuple): 3D center (x, y, z).
        length (float): Side length of cube.

    Returns:
        np.ndarray: Array of (x, y, z) coordinates.
    """
    half_length = length / 2
    x = np.random.uniform(center[0] - half_length, center[0] + half_length, num)
    y = np.random.uniform(center[1] - half_length, center[1] + half_length, num)
    z = np.random.uniform(center[2] - half_length, center[2] + half_length, num)
    return np.column_stack((x, y, z))


if __name__ == "__main__":
    print(gen_settings(np.array([0, 0]), 50))
