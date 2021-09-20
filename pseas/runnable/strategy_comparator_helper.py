"""
Thsi is only a helper to generate data.
It enables to collect data while running a specific strategy.
"""

from numpy import floor
from pseas.test_env import TestEnv
from pseas.strategy import Strategy
from pseas.discrimination.wilcoxon import Wilcoxon

from typing import Callable, List, Dict, Optional, Tuple, Union

from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import wait, ALL_COMPLETED


def __evaluate__(scenario_path: str, distribution: str, strategy: Strategy, 
                 par_penalty: float,  
                 algorithm: int,
                 removed_algorithms: List[int] = [],
                 verbose: bool = False, **kwargs) -> Tuple[Strategy, TestEnv, Dict]:
    env: TestEnv = TestEnv(
        scenario_path, distribution, seed=0, verbose=verbose, par_penalty=par_penalty)
    env.set_removed_algorithms(removed_algorithms)
    stats = {
            "time": [],
            "confidence": [],
            "prediction": [],
            "strategy": [],
            "a_new": [],
            "a_old": [],
            "instances": []
    }
    real = {
        "prediction": [],
        "time": [],
        "a_old": [],
        "instances": []
    }
    to_ratio = lambda x: int(floor(x * 100))
    label: str = strategy.name()
    for cmp in range(env.n_algorithms):
        if cmp == algorithm:
            continue
        state, information, information_has_changed = env.reset((algorithm, cmp))
        if information_has_changed:
            strategy.ready(**information)
        strategy.reset()
        strategy.feed(state)
        last_time_ratio: float = 0
        instances : int = 0
        finished: bool = False
        while instances < env._n_instances:
            state = env.step(strategy.choose_instance())
            strategy.feed(state)
            instances += 1
            #  Update if time changed enough
            time_ratio: float = env.current_time / env.current_max_time
            if to_ratio(last_time_ratio) < to_ratio(time_ratio):
                for i in range(to_ratio(last_time_ratio), to_ratio(time_ratio)):
                        # Update predictions
                        stats["time"].append(i)
                        stats["prediction"].append(
                            strategy.is_better() == env.is_better)
                        stats["strategy"].append(label)
                        stats["a_new"].append(algorithm)
                        stats["a_old"].append(cmp)
                        stats["instances"].append(instances)

                        # Update confidence
                        try:
                            stats["confidence"].append(
                                strategy.get_current_choice_confidence() * 100)
                        except AttributeError:
                            stats["confidence"].append(100)
                last_time_ratio = time_ratio

            if not finished and strategy.get_current_choice_confidence() >= .95:
                if isinstance(strategy._prediction, Wilcoxon) and env.current_instances < 5:
                    continue
                finished = True
                real["a_old"].append(cmp)
                real["prediction"].append(strategy.is_better())
                real["time"].append(env.current_time / env.current_max_time)
                real["instances"].append(env.current_instances)
        env.choose(strategy.is_better())
        # Fill in the rest
        for i in range(to_ratio(last_time_ratio), 101):
                # Update predictions
                stats["time"].append(i)
                stats["strategy"].append(label)
                stats["a_new"].append(algorithm)
                stats["a_old"].append(cmp)
                stats["instances"].append(instances)
                stats["prediction"].append(
                    strategy.is_better() == env.is_better)
                # Update confidence
                try:
                    stats["confidence"].append(
                        strategy.get_current_choice_confidence() * 100)
                except AttributeError:
                    stats["confidence"].append(100)
   
        if not finished:
            finished = True
            real["a_old"].append(cmp)
            real["prediction"].append(strategy.is_better())
            real["time"].append(env.current_time / env.current_max_time)
            real["instances"].append(env.current_instances)
    kwargs["stats"] = stats
    kwargs["real"] = real
    kwargs["a_new"] = algorithm
    return strategy, env, kwargs


def compare(scenario_path: str, 
            strategies: List[Tuple[Union[Strategy, Callable[[None], Strategy]], Dict]],
            distribution: str,
            callback: Callable[[Tuple[Strategy, TestEnv, Dict]], None],
            n_algorithms: int,
            removed_algorithms: List[int] = [],
            par_penalty: float = 1,
            max_workers: Optional[int] = None,
            close_pool: bool = True,
            verbose: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Compare the different startegies on this dataset.
    The stats of each strategy is printed on the standard output

    Parameters:
    -----------
    - scenario_path (str) - file path of the scenario
    - strategies (List[Strategy]) - the list of strategies to compare
    - distribution (str) - the name of the class of the distribution to use to compute the distribution
    - callback (Callable[[Tuple[Strategy, TestEnv, Dict]], None]) - the function that is called on each set of runs after it is terminated.
    - n_algorithms (int) - number of algorithms in this dataset
    - par_penalty (float) - par penalty coefficient. Default: 1.
    - max_workers (Optional[int]) - the number of workers to use. Default: None.
    - verbose (bool) - print info to stdout. Default: False.
    """
    executor = ProcessPoolExecutor(max_workers)
    futures = []
    for algorithm in range(n_algorithms):
        for strategy, kwargs in strategies:
            if algorithm in kwargs.get("a_new_done", []):
                continue
            future = executor.submit(__evaluate__, scenario_path, distribution, strategy.clone(), par_penalty, algorithm, removed_algorithms, verbose, **kwargs)
            future.add_done_callback(callback)
            futures.append(future) 

    if close_pool:
        wait(futures, return_when=ALL_COMPLETED)
