from pseas.discrimination.discrimination import Discrimination
import pseas.truncated_distributions.trunc_cauchy as trunc_cauchy
import pseas.truncated_distributions.trunc_norm as trunc_norm

from typing import Union, Tuple, List, Optional, Callable

import numpy as np

import scipy.stats as st

_CONSTRAINED_ = {
    "cauchy": trunc_cauchy,
    "norm": trunc_norm
}


class DistributionBased(Discrimination):
    """
    Distribution-based discrimination component.
    The bounds on the running times are defined with factors alpha and beta.
    That means that it assumes running times between alpha * t_min and min(cutoff_time, beta * t_max).
    """
    def __init__(self, distribution: str, constrained: bool = True, alpha: float = 0, beta: float = 10, confidence: float = .95) -> None:
        super().__init__()
        self.confidence: float = confidence
        self._constrained: bool = constrained
        self._alpha: float = alpha
        self._beta: float = beta
        self._error_confidence: float = 0
        # Change the way we sum the scale parameter depending on the distribution
        self._sum_scales: Callable[[Union[List[float], np.ndarray]], float] =  sum
        if distribution == "norm":
            self._sum_scales = lambda it: np.sqrt(np.sum(np.square(it)))
        self._distribution: st.rv_continuous = getattr(st, distribution)
        self._dist_name: str = distribution

        name: str = "distribtuion-based " + distribution.title().lower()
        if constrained:
            name += f" constr.[{self._alpha}, {self._beta}]"
        name += f" {self.confidence*100:.0f}%"
        self._name: str = name

    def ready(self, cutoff_time: float, same_class_distributions: np.ndarray, time_bounds: Optional[np.ndarray] = None, **kwargs) -> None:
        # Find timeout
        self._timeout: float = cutoff_time

        self._locs, self._scales = same_class_distributions[:, 0].copy(
        ), same_class_distributions[:, 1].copy()

        if time_bounds is None:
            self._mins: np.ndarray = np.zeros(self._locs.shape[0])
            self._maxs: np.ndarray = np.ones(self._locs.shape[0]) * cutoff_time
        else:
            self._mins: np.ndarray = self._alpha * time_bounds[:, 0]
            self._maxs: np.ndarray = np.minimum(
                self._beta * time_bounds[:, 1], self._timeout)

    def reset(self) -> None:
        self._error_confidence = 0
        self._is_better = False
        self._should_stop: bool = False
    
    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        indices_not_run: List[int] = [
            i for i, time in enumerate(state[0]) if time is None]

        if len(indices_not_run) == 0:
            self._should_stop = True
            self._is_better = np.sum(state[0]) < np.sum(state[1])
            return
        observed_error: float = sum(
            [time - state[1][i] for i, time in enumerate(state[0]) if time is not None])
        loc: float = sum([self._locs[i] - state[1][i]
            for i in indices_not_run])
        scale: float = self._sum_scales([self._scales[i]
            for i in indices_not_run])

        self._current_error: float = observed_error

        # Bounds on the total error (used only if the error model is constrained)
        kwargs = {}
        if self._constrained:
            a: float = sum([self._mins[i] - state[1][i]
                            for i in indices_not_run])
            b: float = sum([self._maxs[i] - state[1][i]
                            for i in indices_not_run])
            kwargs = {"a": a, "b": b}


        distribution = self._distribution
        # Take constraiend distribution
        if self._constrained:
            distribution = _CONSTRAINED_[distribution.name]

        prob_new_is_better: float = distribution.cdf(-observed_error, loc=loc, scale=scale, **kwargs)
        prob_old_is_better: float = 1 - prob_new_is_better

        self._error_confidence: float = max(prob_new_is_better, prob_old_is_better)
        self._is_better = prob_new_is_better > prob_old_is_better
        self._should_stop = self._error_confidence >= self.confidence or len(indices_not_run) == 0

    def get_current_choice_confidence(self) -> float:
        return self._error_confidence

    def should_stop(self) -> bool:
        return self._should_stop

    def is_better(self) -> bool:
        return self._is_better

    def name(self) -> str:
        return self._name


    def clone(self) -> 'DistributionBased':
        return DistributionBased(self._dist_name, self._constrained, self._alpha, self._beta, self.confidence)
