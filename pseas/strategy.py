from abc import ABC, abstractmethod
from typing import Tuple, List, Optional


class Strategy(ABC):

    def ready(self, **kwargs) -> None:
        pass

    def reset(self) -> None:
        """
        Reset the strategy so that it can start anew but with the same ready data.
        """
        pass

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def choose_instance(self) -> int:
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

    @abstractmethod
    def clone(self) -> 'Strategy':
        pass
