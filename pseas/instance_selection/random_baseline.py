from pseas.instance_selection.instance_selection import InstanceSelection

from typing import Tuple, List, Optional

import numpy as np


class RandomBaseline(InstanceSelection):
    """
    Random selection method, also the baseline.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._generator = np.random.default_rng(seed)
        self._seed = seed

    def ready(self, **kwargs) -> None:
        pass

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        not_run_instances: List[int] = [index for index, time in enumerate(state[0]) if time is None]
        if not_run_instances:
            self._next = self._generator.choice(not_run_instances)

    def choose_instance(self) -> int:
        return self._next

    def name(self) -> str:
        return "random"

    def clone(self) -> 'RandomBaseline':
        return RandomBaseline(self._seed)
