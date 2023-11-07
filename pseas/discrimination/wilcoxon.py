import numpy as np
from pseas.discrimination.discrimination import Discrimination

from typing import Tuple, List, Optional

from scipy.stats import wilcoxon

import warnings


class Wilcoxon(Discrimination):
    """
    Wilcoxon based discrimination method.

    Note that the confidence may go over the target confidence before running 5 instances nevertheless the is_done method won't return true before the challenger has been run on 5 instances.

    Parameter:
    ----------
    - confidence: target confidence level
    """

    def __init__(self, confidence: float = 0.95) -> None:
        super().__init__()
        self.confidence: float = confidence

    def ready(self, **kwargs) -> None:
        self._current_confidence = 0

    def reset(self) -> None:
        """
        Reset this scheme so that it can start anew but with the same ready data.
        """
        self._is_done: bool = False
        self._current_confidence = 0

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        instances_done: List[int] = [
            inst for inst, time in enumerate(state[0]) if time is not None
        ]
        # Nothing run = nothing to do
        if len(instances_done) == 0:
            self._current_confidence = 0
            self._is_better = True
            return

        # Everything has been run, we are done
        if len(instances_done) == len(state[0]):
            self._is_done = True
            return

        x1: List[float] = [state[0][i] for i in instances_done]
        x2: List[float] = [state[1][i] for i in instances_done]
        if all([state[0][i] - state[1][i] == 0 for i in instances_done]):
            return
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            _, p_stop = wilcoxon(x1, x2, alternative="two-sided")
            self._current_confidence: float = 1 - p_stop
            self._is_better: bool = np.mean(x1) < np.mean(x2)
            self._is_done = (
                len(instances_done) >= 5 and self._current_confidence >= self.confidence
            ) or self._is_done

    def should_stop(self) -> bool:
        return self._is_done

    def get_current_choice_confidence(self) -> float:
        return self._current_confidence

    def is_better(self) -> bool:
        return self._is_better

    def name(self) -> str:
        return f"Wilcoxon {self.confidence*100:.0f}%"

    def clone(self) -> "Wilcoxon":
        return Wilcoxon(self.confidence)
