from abc import ABC, abstractmethod
from typing import Tuple, List, Optional


class Strategy(ABC):
    """
    Represents a Strategy interface.
    """

    def ready(self, **kwargs) -> None:
        """
        Init a strategy with data given in **kwargs.
        """
        pass

    def reset(self) -> None:
        """
        Reset the strategy so that it can start anew but with the same ready data.
        """
        pass

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        """
        Provides the data for the current state to the strategy which should update its current state.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """
        Return true if the strategy has reached the desired confidence level.
        """
        pass

    @abstractmethod
    def choose_instance(self) -> int:
        """
        Return the instance on which the challenger should be run on next.
        """
        pass

    @abstractmethod
    def get_current_choice_confidence(self) -> float:
        """
        Return the confidence [0;1] in the current choice (is_better).
        """
        pass

    @abstractmethod
    def is_better(self) -> bool:
        """
        Returns the current prediction: true iff it estimates that the challenger is faster than the incumbent.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the strategy.
        """
        pass

    @abstractmethod
    def clone(self) -> 'Strategy':
        """
        Clone this strategy, the cloned strategy has the same parameters but it starts with an empty state (as if no methods were called).
        """
        pass
