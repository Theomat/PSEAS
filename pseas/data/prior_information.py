from typing import Any, List, Dict, Optional, Tuple

import numpy as np

import scipy.stats as st


def initial_guess(distribution_name: str, data: np.ndarray) -> Dict[str, Any]:
    if data.shape[0] == 0:
        return {}
    if distribution_name == "cauchy":
        p25, p50, p75 = np.percentile(data, [25, 50, 75])
        return {
            "loc": p50, 
            "scale": (p75 - p25) / 2
        }
    elif distribution_name == "norm":
        return {
            "loc": np.mean(data),
            "scale": np.std(data)
        }
    return {}

def fit_same_class(distribution_name: str, perf_matrix: np.ndarray) -> np.ndarray:
    distribution = getattr(st, distribution_name)
    prior: np.ndarray = np.zeros(
        (perf_matrix.shape[0], 2), dtype=np.float64)
    for instance in range(perf_matrix.shape[0]):
        data = perf_matrix[instance, :]
        loc, scale = distribution.fit(data, **initial_guess(distribution_name, data))
        prior[instance, 0] = loc
        prior[instance, 1] = scale
    return prior


def resultdict2matrix(results: Dict[str, Dict[str, float]], algorithms: Optional[List[str]]) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Transform a results dictionnary into a performance matrix.

    Parameters:
    -----------
    - results (Dict[str, Dict[str, float]]) - the results dictionnary
    - algorithms (Optional[List[str]]) - the list of algorithms to use to compute the matrix. If None all algorithms are used. Default None.

    Return:
    -----------
    A tuple (perf_matrix, instance2index, algorithm2index).
    perf_matrix (np.ndarray) is the performance matrix, row is the instance index and column is the algorithm index.
    instance2index (Dict[str, int]) a mapping from instance name to row index
    algorithm2index (Dict[str, int]) a mapping from algorithm name to column index
    """
    algorithms = algorithms or results[list(results.keys())[0]].keys()
    num_instances: int = len(results.keys())
    num_algorithms: int = len(algorithms)
    perf_matrix: np.ndarray = np.zeros(
        (num_instances, num_algorithms), dtype=np.float64)
    instance2index: Dict[str, int] = {}
    algorithm2index: Dict[str, int] = {}
    for instance_index, (instance_name, instance_perfs) in enumerate(results.items()):
        instance2index[instance_name] = instance_index
        for algorithm_name, perf in instance_perfs.items():
            if algorithm_name not in algorithms:
                continue
            index: int = len(algorithm2index.keys())
            if algorithm_name in algorithm2index:
                index = algorithm2index[algorithm_name]
            else:
                algorithm2index[algorithm_name] = index
            perf_matrix[instance_index, index] = perf
    return perf_matrix, instance2index, algorithm2index


def compute_all_prior_information(features_dict: Dict[str, np.ndarray], results: Dict[str, Dict[str, float]], algorithms, distribution: str, cutoff_time: float, par_penalty: float) -> Dict[str, Any]:
    perf_matrix, instance2index, _ = resultdict2matrix(
        results, algorithms)

    # Compute time bounds
    time_bounds = np.zeros((perf_matrix.shape[0], 2))
    time_bounds[:, 0] = np.min(perf_matrix, axis=1)
    time_bounds[:, 1] = np.max(perf_matrix, axis=1)

    # Compute feature matrix
    feature_vect_len: int = features_dict[list(
        features_dict.keys())[0]].shape[0]
    features = np.zeros((perf_matrix.shape[0], feature_vect_len), dtype=float)
    for inst, vect in features_dict.items():
        features[instance2index[inst]] = vect

    # Compute same class distributions
    same_class_distributions = fit_same_class(distribution, perf_matrix)

    return {
        "features": features,
        "results": results,
        "perf_matrix": perf_matrix,
        "same_class_distributions": same_class_distributions,
        "distributions": same_class_distributions,
        "time_bounds": time_bounds,
        "cutoff_time": cutoff_time,
        "par_penalty": par_penalty,
    }
