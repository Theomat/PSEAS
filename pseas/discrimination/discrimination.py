from abc import ABC, abstractmethod
from typing import Tuple, List, Optional


class Discrimination(ABC):
    """
    Discrimination method interface.
    Basically method calls to a strategy call the same method if it exists in the discrimination method.
    """

    @abstractmethod
    def ready(self, **kwargs) -> None:
        pass

    def reset(self) -> None:
        """
        Reset this scheme so that it can start anew but with the same ready data.
        """
        pass

    @abstractmethod
    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        pass

    @abstractmethod
    def should_stop(self) -> bool:
        pass

    @abstractmethod
    def get_current_choice_confidence(self) -> float:
        pass

    @abstractmethod
    def is_better(self) -> bool:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def clone(self) -> "Discrimination":
        pass
