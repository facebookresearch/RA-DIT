###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
from typing import Dict, List

from src.data.eval.task.base import Example, GenerationTask
from src.data.eval.utils import (
    answer_level_recall,
    exact_match_score,
    f1_score,
    normalize_answer,
)

# NaturalQuestions example
# {
#     "question": "who's played the most years in the nba",
#     "answers": ["Robert Parish", "Kevin Garnett", "Kevin Willis"],
# }


class NQTask(GenerationTask):
    dataset_dirs = ["nq"]
    metrics = [
        "em",
        "alr",
        "f1",
        "nll",
        "nll_char",
        "nll_token",
        "oracle_em",
        "nll_char_em",
        "nll_token_em",
        "nll_char_f1",
        "nll_token_f1",
    ]
    # fewshot_mode = "first"
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
    eval_file = "test.jsonl"
    question_prefix = "Q:"
    target_prefix = "A:"
    prompt_format = "{question_prefix} {question}\n{target_prefix}"

    # max_text_len = 1536
    max_text_len = 2024
    max_gen_len = 24

    def textify_example(self, example: Example) -> str:
        context = self.get_prompt(example)
        answer = self.get_target(example)
        return f"{context} {answer}"

    def get_prompt(self, example: Example) -> str:
        question = example["question"].capitalize().strip().rstrip("?") + "?"
        prompt = self.prompt_format.format(
            question_prefix=self.question_prefix,
            question=question,
            target_prefix=self.target_prefix,
        )
        return prompt

    def get_target(self, example: Example) -> str:
        answers = example["answers"]
        target = answers[0]  # select first answer as target
        # target = random.choice(answers)
        return target

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = (
            generation.split(self.question_prefix)[0]
            .split(self.target_prefix)[0]
            .split(self.sep)[0]
        )
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = example["answers"]
        sample_metrics = {
            "em": 100 * exact_match_score(prediction, ground_truths, normalize_answer),
            # Answer-Level Recall: Similar to EM, but allows extraneous contents
            # e.g. ALR("The author is Albert Einstein.", "Einstein") == 1.0
            "alr": 100
            * answer_level_recall(prediction, ground_truths, normalize_answer),
            "f1": 100 * f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
