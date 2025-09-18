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
from src.data.eval.utils import exact_match_score, normalize_answer

# Zero-Shot RE example (KILT version)
# {
#     "id": "92bb0167-f709-45ba-bcbd-9d432038f8fe",
#     "input": "Abarema cochleata var. moniliformis [SEP] IUCN conservation status",
#     "output": [{"answer": "vulnerable",
#                 "provenance": [
#                       {"wikipedia_id": "25017687",
#                        "title": "Abarema cochleata var. moniliformis",
#                        ...
#                       }
#                  ]}]
# }


class ZeroShotRETask(GenerationTask):
    dataset_dirs = ["zsre"]
    metrics = [
        "em",
        "nll",
        "nll_char",
        "nll_token",
        "oracle_em",
        "nll_char_em",
        "nll_token_em",
    ]
    fewshot_mode = "first"
    # fewshot_index: List[int] = [0, 462, 818, 1350, 1859] # manual
    fewshot_index: List[int] = [46108, 115346, 60059, 81610, 73107]  # random
    n_fewshot = 5
    sep = "\n\n"
    target_prefix = "A:"
    fewshot_file = "structured_zeroshot-train-kilt-20000.jsonl"
    eval_file = "structured_zeroshot-dev-kilt.jsonl"
    prompt_format = "{context}\n{target_prefix}"
    max_text_len = 1984
    max_gen_len = 16

    def textify_example(self, example: Example) -> str:
        context = self.get_prompt(example)
        answer = self.get_target(example)
        return f"{context} {answer}"

    def get_prompt(self, example: Example) -> str:
        prompt = self.prompt_format.format(
            context=example["input"],
            target_prefix=self.target_prefix,
        )
        return prompt

    def get_target(self, example: Example) -> str:
        answers = example["output"]
        target = answers[0]["answer"]
        return target

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split("\n")[0]
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = [
            example["output"][i]["answer"]
            for i in range(len(example["output"]))
            if "answer" in example["output"][i]
        ]
        assert len(ground_truths) > 0
        sample_metrics = {
            "em": 100 * exact_match_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics


class ZeroShotREQuestionTask(ZeroShotRETask):
    dataset_dirs = ["zsrequestion"]

    def get_prompt(self, example: Example) -> str:
        prompt = self.prompt_format.format(
            context=example["meta"]["template_questions"][0],
            target_prefix=self.target_prefix,
        )
        return prompt
