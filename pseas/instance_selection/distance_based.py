from pseas.instance_selection.instance_selection import InstanceSelection

from typing import Tuple, List, Optional, Callable

import numpy as np


def __compute_distance_matrix__(
    features: np.ndarray, distance: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """
    Computes the distance matrix between the instances.
    It assumes the distance function is symmetric that is d(x,y)=d(y,x) and it assumes d(x, x)=0.

    Parameters:
    -----------
    - features (np.ndarray) - the features of the instances
    - distance (Callable[[np.ndarray, np.ndarray], float]) - a function that given two features compute their distance

    Return:
    -----------
    The distance_matrix (np.ndarray) the distance matrix.
    """
    num_instances: int = features.shape[0]
    distance_matrix: np.ndarray = np.zeros(
        (num_instances, num_instances), dtype=np.float64
    )
    for instance1_index in range(num_instances):
        features1: np.ndarray = features[instance1_index]
        for instance2_index in range(instance1_index + 1, num_instances):
            d: float = distance(features1, features[instance2_index])
            distance_matrix[instance2_index, instance1_index] = d
            distance_matrix[instance1_index, instance2_index] = d
    return distance_matrix


class DistanceBased(InstanceSelection):
    """
    Feature based method that uses the given distance function to the specified exponent.
    Does not correspond to the feature based method in the paper, but it is a parent of that method.

    """

    def __init__(
        self,
        name: str,
        distance: Callable[[np.ndarray, np.ndarray], float],
        exponent: float = 1,
    ) -> None:
        self._distance: Callable[[np.ndarray, np.ndarray], float] = distance
        self._name: str = name + f"-{exponent}"
        self._base_name: str = name
        self._exponent: float = exponent

    def ready(self, features: np.ndarray, distributions: np.ndarray, **kwargs) -> None:
        self._weight_matrix = __compute_distance_matrix__(features, self._distance)
        self._weight_matrix += 1e-10
        np.float_power(self._weight_matrix, -self._exponent, out=self._weight_matrix)
        self._locs = distributions[:, 0]

    def reset(self) -> None:
        self._instances_run: List[int] = []
        self._times = np.tile(self._locs, self._locs.shape[0]).reshape(
            (self._locs.shape[0], -1)
        )

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        instances_not_run: List[int] = [
            i for i, time in enumerate(state[0]) if time is None
        ]

        # Find instance that has been run this turn
        new_instances: List[int] = [
            i
            for i, time in enumerate(state[0])
            if time is not None and i not in self._instances_run
        ]
        if len(new_instances) > 0:
            new_instance: int = new_instances[0]
            self._instances_run.append(new_instance)
            self._times[:, new_instance] = state[0][new_instance]

        if len(instances_not_run) == 0:
            return
        mask: np.ndarray = np.array([time is not None for time in state[0]])
        result_matrix: np.ndarray = self._weight_matrix * self._times
        # set diagonal to 0
        np.fill_diagonal(result_matrix, 0)
        scores: np.ndarray = np.sum(result_matrix, axis=1)
        scores[mask] = np.inf
        self._next = np.argmin(scores)
        assert not mask[self._next]

    def choose_instance(self) -> int:
        return self._next

    def name(self) -> str:
        return self._name

    def clone(self) -> "DistanceBased":
        return DistanceBased(self._base_name, self._distance, self._exponent)
