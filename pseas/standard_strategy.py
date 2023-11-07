from pseas.instance_selection.instance_selection import InstanceSelection
from pseas.discrimination.discrimination import Discrimination
from pseas.strategy import Strategy

from typing import Tuple, List, Optional


class StandardStrategy(Strategy):
    """
    Base strategy that simply combines instance selection and discrimination without any processing of the data.
    """

    def __init__(
        self, instance_selection: InstanceSelection, discrimination: Discrimination
    ) -> None:
        self._instance_selection: InstanceSelection = instance_selection
        self._discrimination: Discrimination = discrimination

    def ready(self, **kwargs) -> None:
        self._instance_selection.ready(**kwargs)
        self._discrimination.ready(**kwargs)

    def reset(self) -> None:
        """
        Reset the strategy so that it can start anew but with the same ready data.
        """
        self._instance_selection.reset()
        self._discrimination.reset()

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        self._instance_selection.feed(state)
        self._discrimination.feed(state)

    def choose_instance(self) -> int:
        return self._instance_selection.choose_instance()

    def is_done(self) -> bool:
        return self._discrimination.should_stop()

    def get_current_choice_confidence(self) -> float:
        return self._discrimination.get_current_choice_confidence()

    def is_better(self) -> bool:
        return self._discrimination.is_better()

    def name(self) -> str:
        return self._instance_selection.name() + " + " + self._discrimination.name()

    def clone(self) -> "StandardStrategy":
        return StandardStrategy(
            self._instance_selection.clone(), self._discrimination.clone()
        )
