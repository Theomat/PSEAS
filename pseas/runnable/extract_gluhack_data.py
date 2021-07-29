from pseas.data import feature_extractor
from pseas.data import result_extractor
from pseas.data import data_transformer
from pseas.data.aslib_scenario import ASlibScenario
import pandas as pd

# =============================================================================
# Argument parsing.
# =============================================================================
import argparse
argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Extract gluhack data from SAT18 run data.")

argument_default_values = {
	"sat18_suffix": 'sat18',
	"dest_suffix": 'gluhack',
	"sat18_path": '../aslib_data/SAT18-EXP',
	"gluhack_path": '../aslib_data/GLUHACK-2018',
}
argument_parser.add_argument('--sat18-suffix',
                             type=str,
                             action='store',
                             default=argument_default_values['sat18_suffix'],
                             help=" (default: 'sat18')"
                             )
argument_parser.add_argument('--dest-suffix',
                             type=str,
                             action='store',
                             default=argument_default_values['dest_suffix'],
                             help=" (default: 'gluhack')"
                             )
argument_parser.add_argument('--sat18-path',
                             type=str,
                             action='store',
                             default=argument_default_values['sat18_path'],
                             help=" (default: '../aslib_data/SAT18-EXP')"
                             )
argument_parser.add_argument('--gluhack-path',
                             type=str,
                             action='store',
                             default=argument_default_values['gluhack_path'],
                             help=" (default: '../aslib_data/GLUHACK-2018')"
                             )
parsed_parameters = argument_parser.parse_args()

sat18_suffix: str = parsed_parameters.sat18_suffix
dest_suffix: str = parsed_parameters.dest_suffix
sat18_path: str = parsed_parameters.sat18_path
gluhack_path: str = parsed_parameters.gluhack_path
# =============================================================================
# Finished parsing
# =============================================================================


def list_hacks():
    scenario = ASlibScenario()
    scenario.read_scenario(gluhack_path)
    scenario.check_data(1)

    features = feature_extractor.from_scenario(scenario)
    results = result_extractor.from_scenario(scenario)

    features, results, _ = data_transformer.prepare_dataset(
        features, results)

    algorithms = list(results[list(features.keys())[0]].keys())
    return algorithms


def find_glucose():
    scenario = ASlibScenario()
    scenario.read_scenario(sat18_path)
    scenario.check_data(1)

    features = feature_extractor.from_scenario(scenario)
    results = result_extractor.from_scenario(scenario)

    features, results, _ = data_transformer.prepare_dataset(
        features, results)

    algorithms = list(results[list(features.keys())[0]].keys())
    hacks = list_hacks()

    indices = []
    glucose = -1
    for i, algo in enumerate(algorithms):
        if algo in hacks:
            indices.append(i)
            if algo == "glucose3.0":
                glucose = i
            print(algo, " perf:", sum([results[inst][algo]
                                       for inst in features.keys()]))

    return glucose, indices


general_df = pd.read_csv(f"./runs_{sat18_suffix}.csv")
general_df = general_df.drop("Unnamed: 0", axis=1)


glucose, others = find_glucose()
# Filter general
general_df = general_df[general_df["a_old"] == glucose]
mask = general_df["a_new"] == others[0]
for other in others[1:]:
    mask |= general_df["a_new"] == other
general_df = general_df[mask]
general_df.to_csv(f"./runs_{dest_suffix}.csv")

detailed_df = pd.read_csv(f"./detailed_runs_{sat18_suffix}.csv")
detailed_df = detailed_df.drop("Unnamed: 0", axis=1)
detailed_df = detailed_df[detailed_df["a_old"] == glucose]
mask = detailed_df["a_new"] == others[0]
for other in others[1:]:
    mask |= detailed_df["a_new"] == other
detailed_df = detailed_df[mask]
detailed_df.to_csv(f"./detailed_runs_{dest_suffix}.csv")
