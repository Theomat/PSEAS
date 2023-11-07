from abc import ABC, abstractmethod
from typing import Tuple, List, Optional


class InstanceSelection(ABC):
    """
    Instance selection method interface.
    Basically method calls to a strategy call the same method if it exists in the instance selection method.
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
    def choose_instance(self) -> int:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def clone(self) -> "InstanceSelection":
        return None
