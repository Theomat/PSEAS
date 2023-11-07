import csv
from collections import defaultdict
from typing import Dict, Iterable

from pseas.data.aslib_scenario import ASlibScenario


def read_results(file: str) -> Dict[str, Dict[str, float]]:
    """
    Read a CSV results file.

    Parameters:
    -----------
    - file (str) - the path of the file to be read

    Return:
    -----------
    A dictionnary containing as keys the name of the instances.
    The value for each instance is a dictionnary with keys the algorithm's name and value the time it took.
    """
    performance_dict: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(file) as fd:
        reader: Iterable = csv.reader(fd)
        next(reader)  # Skip header
        for row in reader:
            instance_name: str = row[0].split("/")[-1][:-4]
            perf_for_instance: Dict[str, float] = performance_dict[instance_name]
            algorithm_name: str = row[1]
            if row[4] == "complete" and (
                row[5] == "SAT-VERIFIED"
                or (row[5] == "UNSAT" and row[7] == "UNSAT-VERIFIED")
            ):
                perf_for_instance[algorithm_name] = float(row[3])
    return performance_dict


def from_scenario(scenario: ASlibScenario) -> Dict[str, Dict[str, float]]:
    """
    Extract the results dictionnary from an ASLibScenario.

    Parameters:
    -----------
    - scenario (ASlibScenario) - the scenario to extract the results from.

    Return:
    -----------
    A dictionnary containing as keys the name of the instances.
    The value for each instance is a dictionnary with keys the algorithm's name and value the time it took.
    """
    results: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(scenario.index_to_instance):
        results[name] = {}
        for algo in scenario.performance_data.columns:
            results[name][algo] = scenario.performance_data[algo].iloc[i]
    return results
