###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from typing import List, Tuple

from src.eval.task.base import ChoiceTask, Example


# FEVER example (KILT version):
# {
#     "id": "137334",
#     "input": "Fox 2000 Pictures released the film Soul Food.",
#     "output": [{"answer": "SUPPORTS", "provenance": [{"wikipedia_id": ...},]}]
# }


class FEVERTask(ChoiceTask):
    dataset_dirs = ["fever"]
    fewshot_mode = "first"
    # fewshot_index: List[int] = [0, 3, 1, 2, 11] # manual
    fewshot_index: List[int] = [18391, 2307, 8097, 33900, 38265]  # random
    n_fewshot = 5
    sep = "\n"
    fewshot_file = "fever-train-kilt-20000.jsonl"
    eval_file = "fever-dev-kilt.jsonl"
    context_format = "Is this statement true? {claim}"
    max_text_len = 2048

    def textify_example(self, example: Example) -> str:
        context = self.context_format.format(claim=example["input"])
        answer = self.choices(example)[example["output"][0]["answer"]]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.context_format.format(claim=example["input"])
        return [(f"{context} {c}", c) for c in ["Yes", "No"]]

    def label(self, example: Example) -> str:
        answer = self.choices(example)[example["output"][0]["answer"]]
        return ["Yes", "No"].index(answer)

    def choices(self, example: Example) -> List[str]:
        mapping = {"SUPPORTS": "Yes", "REFUTES": "No"}
        return mapping
