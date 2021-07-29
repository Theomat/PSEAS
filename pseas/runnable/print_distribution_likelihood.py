from pseas.data import data_transformer, feature_extractor, result_extractor
from pseas.data.aslib_scenario import ASlibScenario

import pandas as pd
import numpy as np
import scipy.stats as st

from typing import List, Tuple
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import wait, ALL_COMPLETED


# =============================================================================
# Argument parsing.
# =============================================================================
import argparse
argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Compute the median log likelihood for each distribution for each dataset.")

argument_default_values = {
	"filename": './dataset_distributions_likelihood.csv',
}
argument_parser.add_argument('-p', '--print-only',
                             action='store_true',
                             dest='plot_only',
                             help="Only print results from file (default: False)"
                             )
argument_parser.add_argument('-f', '--filename',
                             type=str,
                             action='store',
                             default=argument_default_values['filename'],
                             help=" (default: './dataset_distributions_likelihood.csv')"
                             )
parsed_parameters = argument_parser.parse_args()

plot_only: bool = parsed_parameters.plot_only
filename: str = parsed_parameters.filename
# =============================================================================
# Finished parsing
# =============================================================================

datasets = [
    "../aslib_data/BNSL-2016",
    "../aslib_data/SAT18-EXP",
    "../aslib_data/CSP-Minizinc-Time-2016",
    "../aslib_data/SAT20-MAIN",
]
if not plot_only:

    # read from ASlib

    distribs = [
        "cauchy",
        "levy"
    ]

    def __evaluate__(scenario_path: str, distribution: str) -> Tuple[str, str, List[Tuple[float, float]]]:
        scenario = ASlibScenario()
        scenario.read_scenario(scenario_path)
        scenario.check_data(1)

        features = feature_extractor.from_scenario(scenario)
        results = result_extractor.from_scenario(scenario)

        features, results, _ = data_transformer.prepare_dataset(
            features, results)

        dist: st.rv_continuous = getattr(st, distribution)
        output = []
        for _, perf in results.items():
            data = list(perf.values())
            args = dist.fit(data)

            _, pvalue = st.kstest(data, distribution, args=args)
            lklh = np.sum(np.log(dist.pdf(data, *args)))
            output.append((pvalue, lklh))
        return scenario_path, distribution, output

    df = {
        "dataset": [],
        "instance": [],
        "distribution": [],
        "pvalue": [],
        "likelihood": []
    }

    def callback(future):
        dataset, dist, stats = future.result()
        dataset_name = dataset[dataset.rfind("/")+1:]
        for instance, (pval, lklh) in enumerate(stats):
            df["dataset"].append(dataset_name)
            df["instance"].append(instance)
            df["distribution"].append(dist)
            df["pvalue"].append(pval)
            df["likelihood"].append(lklh)

    executor = ProcessPoolExecutor()
    futures = []
    for dataset in datasets:
        for dist_name in distribs:
            future = executor.submit(__evaluate__, dataset, dist_name)
            future.add_done_callback(callback)
            futures.append(future)
    _, not_done = wait(futures, timeout=None, return_when=ALL_COMPLETED)
    assert len(not_done) == 0, "Failed to execute all tasks !"
    df = pd.DataFrame(df)
    df.to_csv(filename)


# Print results in any case
df = pd.read_csv(filename)

df = df.rename(columns={"likelihood": "log-likelihood"})
df_lklh = df.drop(columns=["instance"])

for dataset in datasets:
    dataset_name = dataset[dataset.rfind("/")+1:]

    print("Dataset:", dataset_name)
    print(df_lklh[df_lklh["dataset"] == dataset_name].groupby(
        "distribution").median())
