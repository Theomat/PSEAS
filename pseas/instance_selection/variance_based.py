from pseas.instance_selection.instance_selection import InstanceSelection

from typing import Tuple, List, Optional

import numpy as np


class VarianceBased(InstanceSelection):

    def ready(self, distributions: np.ndarray, **kwargs) -> None:
        locs: np.ndarray = distributions[:, 0]
        scales: np.ndarray = distributions[:, 1]
        self._scores: np.ndarray = scales / locs

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        not_run_mask: np.ndarray = np.array([time is None for time in state[0]])
        self._next: int = np.argmax(self._scores * not_run_mask)

    def choose_instance(self) -> int:
        return self._next

    def name(self) -> str:
        return "variance-based"

    def clone(self) -> 'VarianceBased':
        return VarianceBased()
