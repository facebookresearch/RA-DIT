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


# TriviaQA example
#
# {
#     "question": "Who was President when the first Peanuts cartoon was published?",
#     "answers": [
#         "Presidency of Harry S. Truman",
#         "Hary truman",
#         "Harry Shipp Truman",
#         "Harry Truman's",
#         "Harry S. Truman",
#         "Harry S.Truman",
#         "Harry S Truman",
#         "H. S. Truman",
#         "President Harry Truman",
#         "Truman administration",
#         "Presidency of Harry Truman",
#         "Mr. Citizen",
#         "HST (president)",
#         "H.S. Truman",
#         "Mary Jane Truman",
#         "Harry Shippe Truman",
#         "S truman",
#         "Harry Truman",
#         "President Truman",
#         "33rd President of the United States",
#         "Truman Administration",
#         "Harry Solomon Truman",
#         "Harold Truman",
#         "Harry truman",
#         "H. Truman",
#     ],
#     "target": "Harry Truman",
# }


class TQALSRTask(ChoiceTask):
    dataset_dirs = ["tqa_lsr"]
    metrics = ["nll"]
    fewshot_mode = "index"
    fewshot_index: List[int] = [
        36091,
        39766,
        51344,
        34592,
        15514,
    ]  # random 5 shot downsampled from the 64 shots used by Atlas
    n_fewshot = 5
    sep = "\n"
    fewshot_file = "train.jsonl"
    eval_file = "train.jsonl"
    max_text_len = 2048

    def context(self, example: Example) -> str:
        question = example["question"]
        prompt = f"Question: {question}\nAnswer:"
        return prompt

    def textify_example(self, example: Example) -> str:
        context = self.context(example)
        answer = example["target"]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.context(example)
        return [(f"{context} {ans}", ans) for ans in example["choices"].values()]

    def label(self, example: Example) -> int:
        # all options are GT answers, always return the first one
        return 0
