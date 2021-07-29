from pseas.instance_selection.instance_selection import InstanceSelection

from typing import Tuple, List, Optional

import numpy as np


def __order__(algo_a: int, algo_b: int, ranking: int) -> int:
    for algo in ranking:
        if algo == algo_a:
            return 1
        elif algo == algo_b:
            return -1
    return 0

class RankingBased(InstanceSelection):

    def ready(self, perf_matrix: np.ndarray, **kwargs) -> None:
        algorithms_score: np.ndarray = -np.sum(perf_matrix, axis=0)
        ranking: np.ndarray = np.argsort(algorithms_score)

        scores = []
        for instance in range(perf_matrix.shape[0]):
            instance_ranking: np.ndarray = np.argsort(-perf_matrix[instance])
            f_b: float = 0
            f_g: float = 0
            size_g: int = 0
            for algo_a in range(algorithms_score.shape[0]):
                for algo_b in range(algo_a + 1, algorithms_score.shape[0]):
                    global_order = __order__(algo_a, algo_b, ranking)
                    good = global_order == __order__(algo_a, algo_b, instance_ranking)
                    perf_diff: float = perf_matrix[instance, algo_b] - \
                        perf_matrix[instance, algo_a]
                    if good:
                        size_g += 1
                        if global_order == 1:
                            f_g += -perf_diff
                        else:
                            f_g += perf_diff
                    else:
                        if global_order == 1:
                            f_b += perf_diff
                        else:
                            f_b += -perf_diff
            f_b /= np.median(perf_matrix[instance])
            f_g /= np.median(perf_matrix[instance])
            if size_g == 0:
                f_g = - np.inf
            scores.append((size_g, f_b, f_g, instance))

        scores.sort()
        self._order: np.ndarray = np.array([instance for _, _, _, instance in scores])

    def feed(self, state: Tuple[List[Optional[float]], List[float]]) -> None:
        not_run_mask: np.ndarray = np.array(
            [time is None for time in state[0]])
        for instance in self._order:
            if not_run_mask[instance]:
                self._next = instance
                break

    def choose_instance(self) -> int:
        return self._next

    def name(self) -> str:
        return "ranking-based"

    def clone(self) -> 'RankingBased':
        return RankingBased()
