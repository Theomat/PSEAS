"""
Prints to standatd output the difficulty of all the datasets.
"""


from typing import List
from pseas.data import data_transformer, feature_extractor, result_extractor
from pseas.data.aslib_scenario import ASlibScenario

import numpy as np

datasets = [
    "../aslib_data/CSP-Minizinc-Time-2016",
    "../aslib_data/BNSL-2016",
    "../aslib_data/SAT18-EXP",
    "../aslib_data/SAT20-MAIN",
    "../aslib_data/GLUHACK-2018",
]


for scenario_path in datasets:
    scenario = ASlibScenario()
    scenario.read_scenario(scenario_path)
    scenario.check_data(1)

    features = feature_extractor.from_scenario(scenario)
    results = result_extractor.from_scenario(scenario)

    features, results, _ = data_transformer.prepare_dataset(features, results)

    algorithms = list(results[list(features.keys())[0]].keys())
    instances = list(features.keys())

    print(
        "Dataset:",
        scenario_path,
        " algos:",
        len(algorithms),
        "instances:",
        len(instances),
    )

    perf_algos = np.array(
        [
            sum([results[instance][algo] for instance in instances])
            for algo in algorithms
        ]
    )
    difficulties = []
    top_3_algos = np.argsort(perf_algos)[:3].tolist()
    top_10_algos = np.argsort(perf_algos)[: min(10, perf_algos.shape[0])].tolist()
    total_top3: float = 0
    total_top10: List[float] = []
    median = np.median(perf_algos)
    for i, algo_i in enumerate(algorithms):
        times_i = np.array([results[instance][algo_i] for instance in instances])
        perf_i = np.sum(times_i)
        for dij, algo_j in enumerate(algorithms[i + 1 :]):
            times_j = np.array([results[instance][algo_j] for instance in instances])
            times_j = np.maximum(times_j, 0)
            total_diff_ratio = median / abs(np.sum(times_i) - np.sum(times_j))

            score = total_diff_ratio
            difficulties.append(score)

            if i in top_3_algos and (i + dij) in top_3_algos:
                total_top3 += score
                total_top10.append(score)
            elif i in top_10_algos and (i + dij) in top_10_algos:
                total_top10.append(score)
    print("Difficulty:")
    print("\tmean:", np.mean(difficulties))
    print("\tmedian:", np.median(difficulties))
    print("\tTop-3 mean:", total_top3 / 3)
    print("\tTop-10 mean:", np.mean(total_top10))
    print("\tTop-10 median:", np.median(total_top10))
