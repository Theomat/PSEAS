"""
Same as produce_run_data but with additional parameter k which is the number of top k algorithms to keep.
RUn this script with -h for command line options althoug they are similar to produce_run_data.
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

import pseas.data.result_extractor as result_extractor
import pseas.data.feature_extractor as feature_extractor
import pseas.data.data_transformer as data_transformer
from pseas.data.aslib_scenario import ASlibScenario
from pseas.strategy import Strategy
from pseas.runnable.strategy_comparator_helper import compare
from pseas.standard_strategy import StandardStrategy
from pseas.corrected_timeout_strategy import CorrectedTimeoutStrategy
from pseas.discrimination.subset_baseline import SubsetBaseline
from pseas.discrimination.wilcoxon import Wilcoxon
from pseas.discrimination.distribution_based import DistributionBased
from pseas.instance_selection.random_baseline import RandomBaseline
from pseas.instance_selection.discrimination_based import DiscriminationBased
from pseas.instance_selection.ranking_based import RankingBased
from pseas.instance_selection.variance_based import VarianceBased
from pseas.instance_selection.information_based import InformationBased
from pseas.instance_selection.distance_based import DistanceBased
from pseas.instance_selection.feature_based import FeatureBased

# =============================================================================
# Argument parsing.
# =============================================================================
import argparse

from pseas.test_env import TestEnv

argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Produce run data."
)

argument_default_values: Dict = {
    "output_suffix": "minizinc",
    "save_every": 5,
    "max_workers": None,
    "scenario_path": "../aslib_data/CSP-Minizinc-Time-2016",
    "par": 2,
    "topk": 10,
}
argument_parser.add_argument(
    "-o",
    "--output-suffix",
    type=str,
    action="store",
    default=argument_default_values["output_suffix"],
    help="CSV data filename suffix (default: 'minizinc')",
)
argument_parser.add_argument(
    "--save-every",
    type=int,
    action="store",
    default=argument_default_values["save_every"],
    help="Save data every X time. (default: 5)",
)
argument_parser.add_argument(
    "--top",
    type=int,
    action="store",
    default=argument_default_values["topk"],
    help="Top solvers to consider (default: 10)",
)
argument_parser.add_argument(
    "--max-workers",
    type=int,
    action="store",
    default=argument_default_values["max_workers"],
    help="Max number of processes. (default: None)",
)
argument_parser.add_argument(
    "--scenario-path",
    type=str,
    action="store",
    default=argument_default_values["scenario_path"],
    help=" (default: '../aslib_data/CSP-Minizinc-Time-2016')",
)
argument_parser.add_argument(
    "--par",
    type=int,
    action="store",
    default=argument_default_values["par"],
    help=" (default: 1)",
)
parsed_parameters = argument_parser.parse_args()

output_suffix: str = parsed_parameters.output_suffix
save_every: int = parsed_parameters.save_every
max_workers: int = parsed_parameters.max_workers
scenario_path: str = parsed_parameters.scenario_path
par: int = parsed_parameters.par
top_k: int = parsed_parameters.top_k
# =============================================================================
# Finished parsing
# =============================================================================

# Must not be a lambda function to be picklable
general_filename: str = f"./runs_top{top_k}_{output_suffix}.csv"
detailed_filename: str = f"./detailed_runs_top{top_k}_{output_suffix}.csv"


def norm2_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.linalg.norm(x - y)


discriminators = [
    lambda: DistributionBased("cauchy", confidence=101),
    lambda: Wilcoxon(confidence=101),
    lambda: SubsetBaseline(0.2),
]
selectors = [
    lambda: RandomBaseline(0),
    lambda: DiscriminationBased(1.2),
    lambda: RankingBased(),
    lambda: VarianceBased(),
    lambda: InformationBased(),
    lambda: DistanceBased("norm2-based", norm2_distance),
    lambda: FeatureBased()
]

strategy_makers = [
    lambda i, d: StandardStrategy(i, d),
    lambda i, d: CorrectedTimeoutStrategy(i, d, seed=0)
]

def compute_after_topk(scenario_path, topk) -> List[int]:
    scenario = ASlibScenario()
    scenario.read_scenario(scenario_path)
    scenario.check_data(1)

    features = feature_extractor.from_scenario(scenario)
    results = result_extractor.from_scenario(scenario)

    features, results, _ = data_transformer.prepare_dataset(features, results)

    algorithms = list(results[list(features.keys())[0]].keys())
    instances = list(features.keys())

    perf_algos = np.array(
        [
            sum([results[instance][algo] for instance in instances])
            for algo in algorithms
        ]
    )
    return np.argsort(perf_algos)[topk:].tolist()


# Check if file already exists
original_df_general: Optional[pd.DataFrame] = None
original_df_detailed: Optional[pd.DataFrame] = None
if os.path.exists(general_filename):
    original_df_general = pd.read_csv(general_filename)
    original_df_general = original_df_general.drop("Unnamed: 0", axis=1)

    original_df_detailed = pd.read_csv(detailed_filename)
    original_df_detailed = original_df_detailed.drop("Unnamed: 0", axis=1)
    print("Found existing data, continuing acquisition from save.")


df_general = {
    "y_true": [],
    "y_pred": [],
    "time": [],
    "perf_eval": [],
    "perf_cmp": [],
    "instances": [],
    "par": [],
    "strategy": [],
    "a_new": [],
    "a_old": [],
    "dataset": [],
}


df_detailed = {
    "strategy": [],
    "confidence": [],
    "time": [],
    "instances": [],
    "prediction": [],
    "a_new": [],
    "a_old": [],
    "dataset": [],
}

pbar = tqdm(total=0)
removed_algorithms: List[int] = compute_after_topk(scenario_path, top_k)
print("Removed algos:", removed_algorithms)

def convert(x: int, n_algos: int) -> int:
    allowed_count = -1
    for i in range(n_algos):
        if i not in removed_algorithms:
            allowed_count += 1
            if allowed_count == x:
                return i
    assert False, "Prout"


def callback(future):
    pbar.update(1)

    strat, env, dico = future.result()

    # Fill detailed dataframe
    stats = dico["stats"]
    for k, v in stats.items():
        if k == "strategy":
            for el in v:
                df_detailed["dataset"].append(dico["dataset"])
        for el in v:
            df_detailed[k].append(el)
    # Save detailed dataframe
    if pbar.n % save_every == 0:
        df_tmp = pd.DataFrame(df_detailed)
        if original_df_detailed is not None:
            df_tmp = original_df_detailed.append(df_tmp)
        df_tmp.to_csv(detailed_filename)

    # real data
    real = dico["real"]

    # Fill general dataframe
    for par in range(1, 11):
        env.reevaluate_with_new_par(par)
        for eval, cmp, perf_eval, perf_cmp, y_true, _, _, _ in env.runs():
            df_general["y_true"].append(y_true)
            df_general["perf_eval"].append(perf_eval)
            df_general["perf_cmp"].append(perf_cmp)
            df_general["par"].append(par)
            df_general["strategy"].append(strat.name())
            df_general["a_new"].append(eval)
            df_general["a_old"].append(cmp)
            df_general["dataset"].append(dico["dataset"])

            index: int = -1
            for i, other_cmp in enumerate(real["a_old"]):
                if convert(other_cmp, env._n_algorithms) == cmp:
                    index = i
                    break
            if index < 0:
                print("Fatal error !")
                assert False
            else:
                df_general["time"].append(real["time"][index])
                df_general["instances"].append(real["instances"][index])
                df_general["y_pred"].append(real["prediction"][index])
    # Save general dataframe
    if pbar.n % save_every == 0:
        df_tmp = pd.DataFrame(df_general)
        if original_df_general is not None:
            df_tmp = original_df_general.append(df_tmp)
        df_tmp.to_csv(general_filename)




def run(scenario_path, max_workers, par):
    print()
    env = TestEnv(scenario_path)
    dataset_name: str = scenario_path[scenario_path.strip("/").rfind("/") + 1 :]
    # Generate strategies
    total: int = 0
    strategies: List[Tuple[Strategy, Dict]] = []
    for discriminator in discriminators:
        for selection in selectors:
            for strategy_make in strategy_makers:
                strat = strategy_make(selection(), discriminator())
                dico = {
                    "a_new_done": [],
                    "dataset": dataset_name,
                }
                total += top_k

                if original_df_general is not None:
                    tmp = original_df_general[
                        original_df_general["strategy"] == strat.name()
                    ]
                    tmp = tmp[tmp["dataset"] == dataset_name]
                    dico["a_new_done"] = np.unique(tmp["a_new"].values).tolist()
                    total -= len(dico["a_new_done"])                
                strategies.append([strat, dico])
    pbar.total += total
    compare(
        scenario_path,
        strategies,
        "cauchy",
        callback,
        n_algorithms=top_k,
        verbose=False,
        par_penalty=par,
        max_workers=max_workers,
        close_pool=True,
        removed_algorithms=removed_algorithms
    )


run(scenario_path, max_workers, par)

# Last save
df_tmp = pd.DataFrame(df_detailed)
if original_df_detailed is not None:
    df_tmp = original_df_detailed.append(df_tmp)
df_tmp.to_csv(detailed_filename)
df_tmp = pd.DataFrame(df_general)
if original_df_general is not None:
    df_tmp = original_df_general.append(df_tmp)
df_tmp.to_csv(general_filename)
