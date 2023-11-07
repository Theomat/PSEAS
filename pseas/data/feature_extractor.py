import csv
from typing import Dict, Iterable

from pseas.data.aslib_scenario import ASlibScenario

import numpy as np


def read_features(file: str) -> Dict[str, np.ndarray]:
    """
    Read a CSV feature file.

    Parameters:
    -----------
    - file (str) - the path of the file to be read.

    Return:
    -----------
    A dictionnary containing as keys the name of the instance and as value the features of that instance.
    """
    instance_dict: Dict[str, np.ndarray] = {}
    with open(file) as fd:
        reader: Iterable = csv.reader(fd)
        next(reader)
        for row in reader:
            instance_name: str = row[0].split("/")[-1]
            instance_dict[instance_name] = np.asarray(row[1:], dtype=np.double)
    return instance_dict


def save_features(instance_dict: Dict[str, np.ndarray], file: str):
    """
    Write features to a CSV file.

    Parameters:
    -----------
    - instance_dict (Dict[str, np.ndarray]) - the instances and their features
    - file (str) - the path of the file
    """
    with open(file, "w") as fd:
        writer = csv.writer(fd)
        for instance_name, row in instance_dict.items():
            writer.writerow([instance_name] + row.tolist())


def from_scenario(scenario: ASlibScenario) -> Dict[str, np.ndarray]:
    """
    Extract the features dictionnary from an ASLibScenario.

    Parameters:
    -----------
    - scenario (ASlibScenario) - the scenario to extract the features from.

    Return:
    -----------
    A dictionnary containing as keys the name of the instance and as value the features of that instance.
    """
    features: Dict[str, np.ndarray] = {}
    data: np.ndarray = scenario.feature_data.to_numpy()
    for i, name in enumerate(scenario.index_to_instance):
        features[name] = data[i]
    return features


# ==========================================================================================================
# TESTING
# ==========================================================================================================
if __name__ == "__main__":
    instance_dict: Dict[str, np.ndarray] = read_features("./data/SAT18_features.csv")
    print(
        "read {} instances with {} features each".format(
            len(instance_dict), instance_dict[list(instance_dict.keys())[0]].shape[0]
        )
    )
