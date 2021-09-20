"""
Plot for each dataset two graphs:
    1) Total runtime of algorithm with respect to its ranking as fastest algorithm
    2) Median runtime of instance ordered from lowest to highest

"""


from pseas.data import data_transformer, feature_extractor, result_extractor, prior_information
from pseas.data.aslib_scenario import ASlibScenario

import numpy as np
import matplotlib.pyplot as plt

datasets = [
    "../aslib_data/CSP-Minizinc-Time-2016",
    "../aslib_data/BNSL-2016",
    "../aslib_data/SAT18-EXP",
    "../aslib_data/SAT20-MAIN"
]


for scenario_path in datasets:
    scenario = ASlibScenario()
    scenario.read_scenario(scenario_path)
    scenario.check_data(1)

    features = feature_extractor.from_scenario(scenario)
    results = result_extractor.from_scenario(scenario)

    features, results, _ = data_transformer.prepare_dataset(
        features, results)

    algorithms = list(results[list(features.keys())[0]].keys())
    instances = list(features.keys())

    print("Dataset:", scenario_path, " algos:", len(
        algorithms), "instances:", len(instances))

    perf_matrix = prior_information.resultdict2matrix(results, None)[0]

    perf_algos = np.sum(perf_matrix, axis=0)
    perf_algos = np.sort(perf_algos)
    perf_instances = np.median(perf_matrix, axis=1)
    perf_instances = np.sort(perf_instances)

    plt.subplot(121)
    plt.plot(perf_algos, "x-")
    plt.ylabel("Total Time in s")
    plt.xlabel("Algorithm Rank")
    plt.subplot(122)
    plt.plot(perf_instances, "x-")
    plt.ylabel("Median Time in s")
    plt.xlabel("Instance rank")
    plt.suptitle(scenario_path)
    plt.show()
