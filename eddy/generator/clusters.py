""" Generate autoregressive """
from typing import Union
import numpy as np


def random_sequence(
    start: np.array,
    mu: float,
    temperature: float,
    length: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a random walk from a given starting point.

    Parameters:
        start (np.array): The initial starting point of the sequence.
        mu (float | np.array): The center of the distribution that the step sizes are drawn from.
        temperature (float | np.array): The standard deviation of the distribution that the step sizes are drawn from.
        length (int): The length of the sequence.
        dim (int): The dimension of the vectors.
        rng (np.random.Generator): The numpy random generator.

    Returns:
        (np.array): The sequence of points constituting a random walk.
    """
    sequence = np.zeros((length + 1, dim))
    sequence[0] = start
    for i in range(1, length + 1):
        sequence[i] = sequence[i - 1] + rng.normal(mu, temperature, size=(dim,))
    return sequence


def random_cluster(
    center: np.array,
    cluster_std: np.array,
    n_points: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a random cluster centered at center.

    Parameters:
        center (np.array): The center of the cluster.
        cluster_std (np.array): The standard deviation of the cluster.
        n_points (int): The number of points in the cluster.
        dim (int): The dimension of the points.
        rng (np.random.Generator): Random generator.

    Returns:
        (np.array): Points randomly scattered around the center.
    """
    offsets = rng.normal(0, cluster_std, size=(n_points, dim))
    return center + offsets


def autoregressive_cluster(
    n_steps: int,
    min_points_per_step: int,
    max_points_per_step: int,
    temperature: float,
    n_features: int,
    center: np.array,
    cluster_std: Union[np.array, float],
    rng: np.random.Generator,
) -> tuple[np.array, np.array]:
    """
    Create a cluster with a center that moves in a random walk through feature space.

    Parameters:
        n_steps (int): The number of steps to take.
        min_points_per_step (int): The minimum size of a cluster at a time step.
        max_points_per_step (int): The maximum size of a cluster at a time step.
        temperature (float): Determine how big of a step through feature space the walk can take.
        n_features (int): The dimension of the vectors; i.e. the number of features.
        center (np.array): The initial starting point the cluster.
        cluster_std (Union[np.array, float]): The standard deviation of the distribution that cluster points are drawn
            from.
        rng (np.random.Generator): Random number generator.
    Returns:
        (np.array): The points that constitude the cluster.
        (np.array): The index of the step that a point belongs to.
    """
    max_possible_points = n_steps * max_points_per_step

    points = np.zeros((max_possible_points, n_features))
    steps = np.zeros(max_possible_points, dtype=int)

    current_point = 0
    centers = random_sequence(center, 0, temperature, n_steps, n_features, rng)

    for i, point in enumerate(centers):
        n_points_for_step = rng.integers(min_points_per_step, max_points_per_step)
        cluster = random_cluster(point, cluster_std, n_points_for_step, n_features, rng)
        points[current_point : current_point + n_points_for_step] = cluster
        steps[current_point : current_point + n_points_for_step] = i
        current_point += n_points_for_step

    points = points[:current_point]
    steps = steps[:current_point]

    return points, steps
