from pseas.data.prior_information import initial_guess
from pseas.instance_selection.instance_selection import InstanceSelection
from pseas.discrimination.discrimination import Discrimination
from pseas.strategy import Strategy

from typing import Tuple, List, Optional

import numpy as np
import scipy.stats as st

class CorrectedTimeoutStrategy(Strategy):
    """
    Parameters:
    -----------
    - precision (float) - the squared error threshold between parameters of two consecutive iteration to stop. Default: 1.
    - sampling_factor (int) - parameter M in paper.
    - distribution (str) - the distribtuion used to fit the data
    """

    def __init__(self, instance_selection: InstanceSelection, discrimination: Discrimination,
                 precision: float = 1, sampling_factor: int = 40,
                 distribution: str = "cauchy", seed: Optional[int] = None) -> None:
        self._instance_selection: InstanceSelection = instance_selection
        self._discrimination: Discrimination = discrimination
        self._loc_precision: float = precision
        self._scale_precision: float = precision
        self._sampling_factor: int = sampling_factor
        self._generator: np.random.Generator = np.random.default_rng(seed)
        self._distribution: str = distribution
        self._seed = seed

    def ready(self, perf_matrix: np.ndarray, cutoff_time: float, same_class_distributions: np.ndarray, distributions: np.ndarray, **kwargs) -> None:
        perf_matrix = perf_matrix.copy()

        distributions: np.ndarray = distributions.copy()
        distribution_name: str = self._distribution
        dist: st.rv_continuous = getattr(st, distribution_name)
        timeouts_masks: np.ndarray = perf_matrix >= cutoff_time
        # Select instances that are not only timeouts and that have at least 1 timeout
        instances_to_fit: np.ndarray = np.logical_and(np.sum(1 - timeouts_masks, axis=1) > 0, np.sum(timeouts_masks, axis=1) > 0)

        # Instances that have only timeouts
        instances_with_only_timeouts_mask: np.ndarray = np.sum(
            1 - timeouts_masks, axis=1) > 0
        distributions[instances_with_only_timeouts_mask, 0] = cutoff_time
        distributions[instances_with_only_timeouts_mask, 1] = 1e-10

        # First pass without timeouts
        for instance in range(perf_matrix.shape[0]):
            if not instances_to_fit[instance]:
                continue
            data = perf_matrix[instance, :]
            # Remove timeouts
            data = data[data < cutoff_time]
            loc, scale = dist.fit(data, **initial_guess(distribution_name, data))
            distributions[instance, 0] = loc
            distributions[instance, 1] = scale

        A: int = self._sampling_factor
        instance_converged: np.ndarray = np.ones(timeouts_masks.shape[0], dtype=bool)
        instance_converged[instances_to_fit] = False
        converged: bool = np.sum(instance_converged) == instance_converged.shape[0]
        iterations:int = 0
        while not converged:
            for instance in range(perf_matrix.shape[0]):
                if not instances_to_fit[instance] or instance_converged[instance]:
                    continue
                data = perf_matrix[instance, :]
                # Sample new data
                to_sample: int = np.sum(timeouts_masks[instance, :])
                random_data = dist.rvs(loc=distributions[instance, 0], scale=distributions[instance, 1],
                                       size=to_sample * A + iterations, random_state=self._generator)
                quantiles: List[float] = [(i + 1) / (to_sample + 1) for i in range(to_sample)]
                replacement_data = np.quantile(random_data, quantiles)
                # Keep resampling bounded
                replacement_data = np.minimum(replacement_data, cutoff_time * 10)
                # Fit distribution
                data[timeouts_masks[instance, :]] = replacement_data
                loc, scale = dist.fit(data, loc=distributions[instance, 0], scale=distributions[instance, 1])
                # Compute error
                loc_mse = (loc - distributions[instance, 0])**2
                scale_mse = (scale - distributions[instance, 1])**2
                # Store new parameters
                distributions[instance, 0] = loc
                distributions[instance, 1] = scale
                if loc_mse < self._loc_precision and scale_mse < self._scale_precision:
                    instance_converged[instance] = True
                    converged = np.sum(instance_converged) == instance_converged.shape[0]
                    # Last resampling
                    random_data = dist.rvs(loc=distributions[instance, 0], scale=distributions[instance, 1],
                                           size=to_sample * A, random_state=self._generator)
                    replacement_data = np.quantile(random_data, quantiles)
                    # Keep resampling bounded
                    replacement_data = np.minimum(
                        replacement_data, cutoff_time * 10)
                    # Fit distribution
                    data[timeouts_masks[instance, :]] = replacement_data
            iterations += 1
        self._instance_selection.ready(
            perf_matrix=perf_matrix, same_class_distributions=same_class_distributions, 
            distributions=distributions, cutoff_time=cutoff_time, **kwargs)
        self._discrimination.ready(
            perf_matrix=perf_matrix, same_class_distributions=same_class_distributions,
            distributions=distributions, cutoff_time=cutoff_time, **kwargs)

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
        return self._instance_selection.name() + " + " + self._discrimination.name() + f" + CT"

    def clone(self) -> 'CorrectedTimeoutStrategy':
        return CorrectedTimeoutStrategy(self._instance_selection.clone(), 
                                        self._discrimination.clone(), 
                                        self._loc_precision, 
                                        self._sampling_factor, 
                                        self._distribution, 
                                        self._seed)
