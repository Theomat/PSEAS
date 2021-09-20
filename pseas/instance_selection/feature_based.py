from pseas.instance_selection.distance_based import DistanceBased

import numpy as np
from scipy import optimize


def __find_weights__(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    instances: int = x.shape[0]
    features: int = x.shape[1]

    qty: int = int(instances * (instances - 1) / 2)

    dx: np.ndarray = np.zeros((qty, features))
    dy: np.ndarray = np.zeros((qty,))

    # Compute dataset
    index: int = 0
    for i in range(instances):
        for j in range(i + 1, instances):
            dx[index] = x[i] - x[j]
            dy[index] = y[i] - y[j]
            index += 1
    np.square(dx, out=dx)
    np.abs(dy, out=dy)
    # np.square(dy, out=dy)

    # weights = argmin_w_i (norm [w_i (x_i -x'_i)]_i - |y - y'|)^2
    weights, residual = optimize.nnls(dx, dy)
    return np.sqrt(weights)


class FeatureBased(DistanceBased):
    """
    Feature based selection method, it is an instance of the distance based method.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__("feature-based", self._dynamic_distance, *args, **kwargs)

    def _dynamic_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.linalg.norm(self._weights * (x1 - x2))

    def ready(self, features: np.ndarray, perf_matrix: np.ndarray, **kwargs) -> None:
        # Find optimal distance function
        y: np.ndarray = np.median(perf_matrix, axis=1)
        self._weights: np.ndarray = __find_weights__(features, y)
        super().ready(features=features, perf_matrix=perf_matrix, **kwargs)

    def clone(self) -> 'FeatureBased':
        return FeatureBased(self._exponent)
