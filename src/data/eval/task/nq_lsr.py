###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from typing import Dict, List, Tuple

from src.data.eval.task.base import ChoiceTask, Example

# This is a dummy task used for generating LM supervised retriever (LSR) training data for NQ
# This is achieved by converting NQ to a choice task, where each option is a GT answer,
# in order to compute LM likelihood of the answer given each retrieved support doc.

# NaturalQuestions example
# {
#     "question": "who's played the most years in the nba",
#     "answers": ["Robert Parish", "Kevin Garnett", "Kevin Willis"],
#     "choices": {"A": "Robert Parish", "B": "kevin Garnett", "C": "Kevin Willis"},
#     "answer": "A",
# }


class NQLSRTask(ChoiceTask):
    dataset_dirs = ["nq_lsr"]
    metrics = ["nll"]
    fewshot_mode = "index"
    fewshot_index: List[int] = [
        7162,
        13453,
        27144,
        14489,
        38094,
    ]  # random 5 shot downsampled from the 64 shots used by Atlas
    n_fewshot = 5
    sep = "\n"
    fewshot_file = "train.jsonl"
    eval_file = "train.jsonl"
    max_text_len = 2048

    def context(self, example: Example) -> str:
        question = example["question"]
        # choices = example["choices"]
        # choice_str = "\n".join([f"{k}. {v}" for k,v in choices.items()])
        prompt = f"Question: {question}\nAnswer:"
        return prompt

    def textify_example(self, example: Example) -> str:
        context = self.context(example)
        answer = example["choices"][example["answer"]]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.context(example)
        return [(f"{context} {ans}", ans) for ans in example["choices"].values()]

    def label(self, example: Example) -> int:
        # all options are GT answers, always return the first one
        return 0
