from pseas.instance_selection.instance_selection import InstanceSelection

from typing import Tuple, List, Optional, Callable, Union

import numpy as np

import scipy.stats as st
import scipy.integrate as integrate
from scipy.special import rel_entr


def __global_error_parameters__(self, state) -> Tuple[float, float, float]:
    indices_not_run: List[int] = [i for i, time in enumerate(state[0]) if time is None]
    observed_error: float = sum(
        [time - state[1][i] for i, time in enumerate(state[0]) if time is not None]
    )
    loc: float = sum([self._locs[i] - state[1][i] for i in indices_not_run])
    scale: float = self._sum_scales([self._scales[i] for i in indices_not_run])
    return observed_error, loc, scale


def __instance_parameters__(
    self, state, loc: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices_not_run: List[int] = [
        i for i, time in enumerate(state[0]) if time is None and self._scales[i] > 1
    ]

    # Scale parameters computed without each instance
    scale_wo_instance: np.ndarray = np.zeros(
        len(indices_not_run), dtype=np.float64
    )  # parameter for p(X|Y=y)
    for j, i in enumerate(indices_not_run):
        scale_wo_instance[j] = max(
            1e-20,
            self._sum_scales([self._scales[k] for k in indices_not_run if i != k]),
        )
    # Expected error for each instance
    exp_err_for_instance: np.ndarray = np.array(
        [
            -x
            for i, x in enumerate(state[1])
            if state[0][i] is None and self._scales[i] > 1
        ]
    )
    for j, i in enumerate(indices_not_run):
        exp_err_for_instance[j] += self._locs[i]
    # Expected error without each instance
    expected_error_wo_instance: np.ndarray = loc - exp_err_for_instance
    # Scale of each instance
    inst_scales: np.ndarray = np.zeros_like(expected_error_wo_instance)
    for j, i in enumerate(indices_not_run):
        inst_scales[j] = max(1e-20, self._scales[i])
    return (
        expected_error_wo_instance,
        scale_wo_instance,
        exp_err_for_instance,
        inst_scales,
    )


# Goal: Compute the expected gain of information for running instance i, IG_i
# Y = error for instance i
# Y is in |R

# IG_i = E~y [ InfoGain(X | Y=y) ]
# InfoGain(X | Y=y) = D( p(X|Y=y) || p(X) )
# IG_i = E~y[ D( p(X|Y=y) || p(X) ) ]


def __compute_information_relative_to_decision__(self, state, k: float = 5):
    # Here:
    # X = Etot < 0
    # X has 2 values either True or False
    # IG_i = int_y p(Y=y) sum_x p(X=x|Y=y) log (p(X=x|Y=y)/p(X=x))
    pdf = self._distribution.pdf
    cdf = self._distribution.cdf

    observed_error, loc, scale = __global_error_parameters__(self, state)
    (
        expected_error_wo_instance,
        scale_wo_instance,
        inst_locs,
        inst_scales,
    ) = __instance_parameters__(self, state, loc)

    # P(Y=y)
    def prob_y(y: np.ndarray) -> np.ndarray:
        return pdf(y, loc=inst_locs, scale=inst_scales)

    # p(X=0) <=> P(Etot < 0)
    prob_x_is_zero: float = cdf(-observed_error, loc=loc, scale=scale)

    # P(X=0|Y=y) <=> P(Etot < 0 | E_i = e_i)
    def prob_x_is_zero_knowing_y(y: np.ndarray) -> np.ndarray:
        return cdf(
            -observed_error - y, loc=expected_error_wo_instance, scale=scale_wo_instance
        )

    # IG_i = int_y f(y)
    # Odd number of points => even number of intervals => Simpson works well
    y: np.ndarray = np.linspace(
        inst_locs - k * inst_scales, inst_locs + k * inst_scales, num=101
    )
    x_is_zero = prob_x_is_zero
    x_is_zero_knowing_y = prob_x_is_zero_knowing_y(y)
    # sum_x p(X=x|Y=y) log (p(X=x|Y=y)/p(X=x))
    info_gained: np.ndarray = np.maximum(
        rel_entr(x_is_zero_knowing_y, x_is_zero)
        + rel_entr(1 - x_is_zero_knowing_y, 1 - x_is_zero),
        0,
    )
    # P(Y=y)
    prob = prob_y(y)
    # IG_i = int_y p(Y=y) sum_x p(X=x|Y=y) log (p(X=x|Y=y)/p(X=x))
    expected_info: np.ndarray = integrate.simpson(prob * info_gained, x=y, axis=0)
    # Quad_vec is far too slow compared to our method and provides insignificant gain <= 0.3%
    return expected_info


class InformationBased(InstanceSelection):

    """
    Information based selection method.
    """

    def __init__(self, distribution: str = "cauchy") -> None:
        super().__init__()
        self._name: str = "information-based"
        self._distribution_name: str = distribution
        self._distribution: st.rv_continuous = getattr(st, distribution)
        # Change the way we sum the scale parameter depending on the distribution
        self._sum_scales: Callable[[Union[List[float], np.ndarray]], float] = sum
        if distribution == "norm":
            self._sum_scales = lambda it: np.sqrt(np.sum(np.square(it)))
        self.__compute_information__ = __compute_information_relative_to_decision__

    def ready(self, distributions: np.ndarray, **kwargs) -> None:
        self._scores: np.ndarray = np.zeros(distributions.shape[0], dtype=float)
        self._locs = distributions[:, 0]
        self._scales = distributions[:, 1]

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        score_mask: np.ndarray = np.array(
            [time is None and self._scales[i] > 1 for i, time in enumerate(state[0])]
        )
        not_run_mask: np.ndarray = np.array([time is None for time in state[0]])
        if not np.any(not_run_mask):
            return
        # Update scores
        self._scores[:] = -1e10
        # Select not run before a run one
        self._scores[not_run_mask] = np.maximum(self._scales[not_run_mask], 0)
        max_translation: float = max(np.max(self._scales[not_run_mask]), 0)
        self._scores[score_mask] = (
            self.__compute_information__(self, state) / self._locs[score_mask]
            + max_translation
        )
        self._next = np.argmax(self._scores)
        assert not_run_mask[self._next], "Best score is an instance already chosen !"

    def choose_instance(self) -> int:
        return self._next

    def name(self) -> str:
        return self._name

    def clone(self) -> "InformationBased":
        return InformationBased(self._distribution_name)
