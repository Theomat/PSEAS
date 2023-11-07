from pseas.data.prior_information import (
    compute_all_prior_information,
    resultdict2matrix,
)
from pseas.data.aslib_scenario import ASlibScenario
import pseas.data.data_transformer as data_transformer
import pseas.data.feature_extractor as feature_extractor
import pseas.data.result_extractor as result_extractor

from enum import Enum
from typing import Generator, List, Tuple, Optional, Dict, Union

import numpy as np


class ResetChoice(Enum):
    """Defines the type of reset for an environement."""

    RESET_BEST = 0
    """Compare against the best performing algorithm."""
    RESET_RANDOM = 1
    """Compare against a random algorithm."""


class TestEnv:
    """
    A test environment to measure the performance of a strategy.

    Parameters:
    -----------
    - seed (Optional[int]) - the seed to use. Default: None.
    """

    def __init__(
        self,
        scenario_path: str,
        distribution: str = "cauchy",
        par_penalty: float = 1,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self._generator = np.random.default_rng(seed)
        self.par_penalty: float = par_penalty

        scenario = ASlibScenario()
        scenario.read_scenario(scenario_path)
        scenario.check_data()

        features = feature_extractor.from_scenario(scenario)
        results = result_extractor.from_scenario(scenario)

        features, results, maxi_cutoff_time = data_transformer.prepare_dataset(
            features, results
        )
        # print("Practical:", cutoff_time," theorical:", scenario.algorithm_cutoff_time)
        # used_par = int(np.round(maxi_cutoff_time / scenario.algorithm_cutoff_time))

        cutoff_time = par_penalty * scenario.algorithm_cutoff_time

        self._features = features
        self._results = {
            k1: {
                k2: t if t < scenario.algorithm_cutoff_time else cutoff_time
                for k2, t in v.items()
            }
            for k1, v in results.items()
        }
        self._distribution: str = distribution
        self._cutoff_time: float = cutoff_time

        self._n_instances: int = len(features.keys())
        self._index2algo: List[str] = list((list(results.values())[0]).keys())
        self._n_algorithms: int = len(self._index2algo)

        self._removed_algorithms: List[int] = []
        self._removed_algorithms_has_changed: bool = False

        if verbose:
            print(
                "Using",
                self._n_instances,
                "instances with",
                self._n_algorithms,
                "algorithms.",
            )
            print("Cutoff time=", cutoff_time)
        # stats
        self._correct: List[bool] = []
        self._time_ratio: List[float] = []
        self._choices: List[int] = []
        self._history: List[List] = []

    @property
    def action_space(self) -> List[int]:
        """
        The list of possible actions that can be taken.
        """
        return list(range(self._n_instances))

    def legal_moves(self) -> List[int]:
        """
        Get the list of legal actions in the current state.

        Return:
        --------
        A list of valid actions.
        """
        return [x for x in self.action_space if not self._done[x]]

    def __state__(self) -> Tuple[List[Optional[float]], List[float]]:
        times: List[Optional[float]] = [
            self._evaluating_times[i] if self._done[i] else None
            for i in range(self._n_instances)
        ]
        times_cmp: List[float] = [
            self._comparing_times[i] for i in range(self._n_instances)
        ]
        return times, times_cmp

    def reset(
        self, choice: Union[ResetChoice, Tuple[int, int]] = ResetChoice.RESET_RANDOM
    ) -> Tuple[Tuple[List[Optional[float]], List[float], bool], Dict]:
        """
        Reset the current state of the environment.

        Parameters:
        -----------
        - choice (ResetChoice or (challenger_index, incumbent_index)) - the choice type of algorithm to be evaluating

        Return:
        -----------
        A tuple ((my_times, times_comparing), information, information_has_changed).
        my_times (List[Optional[float]]) is a list containing the times the algorithm took on the instances this algorithm was run on.
        If the algorithm wasn't run on a problem it is replaced by None.
        times_comparing (List[float]) is a list containing the times the algorithm we are comparing against took on the instances.
        information (Dict) is the data to pass to the ready function to the strategy
        """
        if isinstance(choice, ResetChoice):
            # First choose evaluating algorithm
            evaluating: int = self._generator.integers(0, self._n_algorithms)
            # Then choose algorithm to compare to
            comparing: int = evaluating
            if choice is ResetChoice.RESET_BEST:
                comparing = None  # we do it later
            else:
                while comparing == evaluating:
                    comparing = self._generator.integers(0, self._n_algorithms)
        else:
            if self._removed_algorithms:

                def convert(x: int) -> int:
                    allowed_count = -1
                    for i in range(self._n_algorithms):
                        if i not in self._removed_algorithms:
                            allowed_count += 1
                            if allowed_count == x:
                                return i
                    print("Fatal error: chosen", x, "while max was:", allowed_count)
                    assert False

                evaluating: int = convert(choice[0])
                comparing: int = convert(choice[1])
            else:
                evaluating: int = choice[0]
                comparing: int = choice[1]

        eval_name: str = self._index2algo[evaluating]

        information_has_changed: bool = True
        # Do as if evaluating never existed in data
        if len(self._history) > 0 and evaluating == self._history[-1][0]:
            # Recompute if removed algorithms changed
            information_has_changed = self._removed_algorithms_has_changed
            self._removed_algorithms_has_changed = False

        # Recompute information
        if information_has_changed:
            removed_names = [self._index2algo[x] for x in self._removed_algorithms]
            results = {
                instance: {
                    algo: time
                    for algo, time in times.items()
                    if eval_name != algo and algo not in removed_names
                }
                for instance, times in self._results.items()
            }
            information = compute_all_prior_information(
                self._features,
                results,
                None,
                self._distribution,
                self._cutoff_time,
                self.par_penalty,
            )
            # Get its times
            self._evaluating_times: np.ndarray = np.array(
                [
                    self._results[instance][eval_name]
                    for instance in self._features.keys()
                ]
            )
            self._last_info = information
        else:
            information = self._last_info

        # we do it now that we have the data
        if comparing is None:
            comparing = np.argmin(np.sum(information["perf_matrix"], axis=0))

        comparing_name: str = self._index2algo[comparing]
        self._comparing_times: np.ndarray = np.array(
            [
                self._results[instance][comparing_name]
                for instance in self._features.keys()
            ]
        )

        self._history.append([evaluating, comparing, False])

        # Assign data
        self._done: np.ndarray = np.zeros(self._n_instances, dtype=np.bool_)
        return self.__state__(), information, information_has_changed

    def set_removed_algorithms(self, removed: List[int]):
        """
        Remove algorithms out of the dataset and act (even after reset) as if they were not in the dataset.
        Must be done just before a reset to behave corretly.
        """
        if removed != self._removed_algorithms:
            self._removed_algorithms = removed
            self._removed_algorithms_has_changed = True

    def choose(self, better: bool):
        """
        Choose wether this algorithm is better or not than the one it's being compared to.
        Once you choose you should reset the environment.

        Parameters:
        -----------
        - better (bool) - indicates whether this algorithm is strictly better or not thant the one it's being compared to
        """
        self._correct.append(self.is_better == better)
        self._time_ratio.append(self.current_time / self.current_max_time)
        self._choices.append(np.sum(self._done))
        self._history[-1][-1] = better

    def reevaluate_with_new_par(self, new_par: float):
        self.par_penalty = new_par
        for index, (eval, cmp, better) in enumerate(self._history):
            eval_name: str = self._index2algo[eval]
            comparing_name: str = self._index2algo[cmp]

            evaluating_times: np.ndarray = np.array(
                [
                    self._results[instance][eval_name]
                    for instance in self._features.keys()
                ]
            )
            comparing_times: np.ndarray = np.array(
                [
                    self._results[instance][comparing_name]
                    for instance in self._features.keys()
                ]
            )
            penalty_eval: float = np.sum(evaluating_times >= self._cutoff_time) * (
                self.par_penalty - 1
            )
            penalty_comparing: float = np.sum(comparing_times >= self._cutoff_time) * (
                self.par_penalty - 1
            )
            is_better: bool = (
                np.sum(evaluating_times) + penalty_eval
                < np.sum(comparing_times) + penalty_comparing
            )
            self._correct[index] = better == is_better

    def step(self, instance: int) -> Tuple[List[Optional[float]], List[float]]:
        """
        Choose the next problem.

        Parameters:
        -----------
        - instance (int) - the instance on which to run the algorithm

        Return:
        ----------
        A tuple (my_times, times_comparing).
        my_times (List[float]) is a list containing the times the algorithm took on the instances this algorithm was run on.
        If the algorithm wasn't run on a problem it is replaced by None.
        times_comparing (List[float]) is a list containing the times the algorithm we are comparing against took on the instances.
        """
        assert not self._done[
            instance
        ], f"Instance {instance} was already chosen ! Choose in: {self.legal_moves()}"
        self._done[instance] = True
        return self.__state__()

    @property
    def n_algorithms(self) -> int:
        """
        Number of algorithms in the current dataset.
        """
        return self._n_algorithms - len(self._removed_algorithms)

    @property
    def is_better(self) -> bool:
        """
        Return true iff the challenger is better than the incumbent.
        """
        penalty_eval: float = np.sum(self._evaluating_times >= self._cutoff_time) * (
            self.par_penalty - 1
        )
        penalty_comparing: float = np.sum(
            self._comparing_times >= self._cutoff_time
        ) * (self.par_penalty - 1)
        return (
            np.sum(self._evaluating_times) + penalty_eval
            < np.sum(self._comparing_times) + penalty_comparing
        )

    @property
    def current_time(self) -> float:
        """
        Total time used so far by the challenger.
        """
        return sum(
            [
                self._evaluating_times[i]
                for i in range(self._n_instances)
                if self._done[i]
            ]
        )

    @property
    def current_instances(self) -> int:
        """
        Number of instances on which the challenger has been executed.
        """
        return np.sum(self._done)

    @property
    def current_comparing_max_time(self) -> float:
        """
        Total time it would take to run the incumbent on all instances.
        """
        return np.sum(self._comparing_times)

    @property
    def current_max_time(self) -> float:
        """
        Total time it would take to run the challenger on all instances.
        """
        return np.sum(self._evaluating_times)

    @property
    def score(self, estimator=np.median) -> float:
        correct = estimator(self._correct)
        time_ratio = estimator(self._time_ratio)
        return (correct - 0.5) * 2 * (1 - time_ratio)

    def stats(self, estimator=np.median) -> Tuple[float, float, float]:
        return (
            estimator(self._correct),
            estimator(self._time_ratio),
            estimator(self._choices),
        )

    def runs(
        self,
    ) -> Generator[Tuple[int, int, float, float, bool, bool, float, int], None, None]:
        for index, (eval, cmp, better) in enumerate(self._history):
            eval_name: str = self._index2algo[eval]
            comparing_name: str = self._index2algo[cmp]

            evaluating_times: np.ndarray = np.array(
                [
                    self._results[instance][eval_name]
                    for instance in self._features.keys()
                ]
            )
            comparing_times: np.ndarray = np.array(
                [
                    self._results[instance][comparing_name]
                    for instance in self._features.keys()
                ]
            )
            penalty_eval: float = np.sum(evaluating_times >= self._cutoff_time) * (
                self.par_penalty - 1
            )
            penalty_comparing: float = np.sum(comparing_times >= self._cutoff_time) * (
                self.par_penalty - 1
            )

            perf_eval: float = np.sum(evaluating_times) + penalty_eval
            perf_cmp: float = np.sum(comparing_times) + penalty_comparing
            is_better: bool = perf_eval < perf_cmp

            yield eval, cmp, perf_eval, perf_cmp, is_better, better, self._time_ratio[
                index
            ], self._choices[index]

    def get_total_perf_matrix(self) -> np.ndarray:
        return resultdict2matrix(self._results, None)[0]
