from numpy import ceil
from pseas.discrimination.discrimination import Discrimination
from typing import Tuple, List, Optional


class SubsetBaseline(Discrimination):

    def __init__(self, ratio:float) -> None:
        super().__init__()
        self._ratio: float = ratio

    def ready(self, **kwargs) -> None:
        pass

    def reset(self) -> None:
        """
        Reset this scheme so that it can start anew but with the same ready data.
        """
        self._is_done = False
        self._confidence = 0

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        self._time: float = sum([x for x in state[0] if x is not None])
        self._is_better: bool = self._time < sum(
            [t for x, t in zip(state[0], state[1]) if x is not None])

        todo: int = ceil(len(state[0]) * self._ratio)
        done: int = len([x for x in state[0] if x is not None])
        self._is_done: bool = done >= todo
        self._confidence: float = 1 if self._is_done else 0

    def should_stop(self) -> bool:
        return self._is_done

    def get_current_choice_confidence(self) -> float:
        return self._confidence

    def is_better(self) -> bool:
        return self._is_better

    def name(self) -> str:
        return f"subset {self._ratio * 100:.2f}%"

    def clone(self) -> 'SubsetBaseline':
        return SubsetBaseline(self._ratio)

