from pseas.instance_selection.instance_selection import InstanceSelection
from typing import Tuple, List, Optional

import numpy as np


class DiscriminationBased(InstanceSelection):
    """
    Discrimination based method based on dominance of algorithms.

    Parameter:
    ----------
    - rho: the domination ratio score = #{ time(algo)/time(best algo) <= rho } / expected_time
    """

    def __init__(self, rho: float) -> None:
        super().__init__()
        self._rho : float = rho

    def ready(self, distributions: np.ndarray, perf_matrix: np.ndarray, **kwargs) -> None:
        locs: np.ndarray = distributions[:, 0]
        self._scores = np.count_nonzero(perf_matrix > np.repeat(self._rho * np.min(perf_matrix, axis=1), perf_matrix.shape[1]).reshape(perf_matrix.shape[0], -1), axis=1).astype(dtype=float)
        self._scores /= locs
        self._scores += 1e-10

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        not_run_mask: np.ndarray = np.array([time is None for time in state[0]])
        self._next = np.argmax(self._scores * not_run_mask)

    def choose_instance(self) -> int:
        return self._next

    def name(self) -> str:
        return f"{self._rho:.2f}-discrimination-based"

    def clone(self) -> 'DiscriminationBased':
        return DiscriminationBased(self._rho)
