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

# T-REx example (KILT version)
# {
#     "id": "2ad0bad9-73df-40a9-810f-3dcebb124473",
#     "input": "Serge Blisko [SEP] occupation",
#     "output": [{"answer": "politician",
#                 "provenance": [
#                       {"wikipedia_id": "20492519",
#                        "title": "Serge Blisko",
#                        ...
#                       }
#                  ]}]
# }


class TRExTask(GenerationTask):
    dataset_dirs = ["trex"]
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
    fewshot_index: List[int] = [385983, 2174782, 1655279, 505453, 396720]  # random
    n_fewshot = 5
    sep = "\n\n"
    target_prefix = "A:"
    fewshot_file = "trex-train-kilt-20000.jsonl"
    eval_file = "trex-dev-kilt.jsonl"
    prompt_format = "{context}\n{target_prefix}"
    max_text_len = 2032
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
