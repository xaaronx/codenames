import ast
import glob
import json
import logging
import sys
import os
from typing import Type

import numpy as np
import yaml
from tqdm import tqdm

from bot import CodeNamesSolverAlgorithm, MeanIndividualDistance, SummedNearestNeighbour, Guess

sys.path.append(os.path.abspath(os.path.join('..')))

from bot.solver import SolverBuilder
from bot.utils import get_embeddings_glove_style, get_embeddings_paragram_style


def solve_single(solver: SolverBuilder, test_case: dict, threshold: float = .3,
                 algorithm: Type[CodeNamesSolverAlgorithm] = MeanIndividualDistance, n: int = 20) -> list:
    runner = solver.build(threshold=threshold, algorithm=algorithm)
    preds = runner.solve(words_to_hit=test_case["words_to_hit"], n=n)
    for p in preds:
        if isinstance(p, Guess):
            yield p.as_dict()
        else:
            yield p


def get_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)


def get_test_cases(case_glob_path: str) -> list:
    files = glob.glob(case_glob_path)
    cases = []
    for file in files:
        data = ast.literal_eval(json.load(open(file)))
        test_case = list(map(str.lower, data["selected_words"].split(";")))
        answer = data["clue"].lower()
        cases.append({"words_to_hit": test_case, "answer": answer})
    return cases


if __name__ == "__main__":

    path = "config.yaml"
    conf = get_yaml_config(path)
    thresholds = np.linspace(conf["threshold_min"], conf["threshold_max"], conf["n_thresholds"])
    algorithms = [MeanIndividualDistance, SummedNearestNeighbour]

    test_cases_glob_path = os.path.join("..", "data", "test", "*")
    test_cases = get_test_cases(test_cases_glob_path)

    for model_config in conf.get("models"):
        predictions = []
        model_config["embeddings_parser"] = eval(model_config["embeddings_parser"])
        model = SolverBuilder.with_embeddings(**model_config)
        for threshold in tqdm(thresholds):
            for algo in algorithms:
                for case in test_cases:
                    for pred in solve_single(model, case, threshold, algo):
                        try:
                            predictions.append({
                                "model_name": model.method,
                                "threshold": threshold,
                                "algorithm": algo.__name__,
                                "words_to_hit": case["words_to_hit"],
                                "true": case["answer"],
                                "prediction": pred
                            })
                        except Exception as e:
                            print(e)

        with open(f"results_{model.method}.json", "w") as f:
            json.dump(predictions, f)
