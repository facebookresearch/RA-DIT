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

# CoNLL-YAGO example (KILT version)
# {
#     "id": "8bcfcff7-a0c9-4b59-b187-a812f75ad08f_1 EU:0",
#     "input": "EU rejects [START_ENT] German [END_ENT] call to boycott British lamb .  Peter Blackburn  BRUSSELS 1996-08-22  The European ...",
#     "output": [{"answer": "Brussels",
#                 "provenance": [
#                       {"wikipedia_id": "3708",
#                        "title": "Brussels",
#                       }
#                  ]}]
# }


class CoNLLYAGOTask(GenerationTask):
    dataset_dirs = ["conllyago"]
    metrics = [
        "em",
        "nll",
        "nll_char",
        "nll_token",
        "oracle_em",
        "nll_char_em",
        "nll_token_em",
    ]
    fewshot_mode = "index"
    # fewshot_index: List[int] = [0, 14, 114, 506, 2428] # manual
    fewshot_index: List[int] = [13808, 7542, 11119, 10322, 11624]  # random
    n_fewshot = 5
    sep = "\n\n"
    target_prefix = "A:"
    fewshot_file = "aidayago2-train-kilt.jsonl"
    eval_file = "aidayago2-dev-kilt.jsonl"
    prompt_format = "{context}\nOutput the Wikipedia page title of the entity mentioned between [START_ENT] and [END_ENT] in the given text\n{target_prefix}"
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


class CoNLLYAGOTruncTask(CoNLLYAGOTask):
    dataset_dirs = ["conllyagotrunc"]
    truncate_length = 50

    def get_prompt(self, example: Example) -> str:
        prompt = self.prompt_format.format(
            context=self.truncate_string(example["input"]),
            target_prefix=self.target_prefix,
        )
        return prompt

    def truncate_string(self, context: str) -> str:
        start_str = "[START_ENT]"
        end_str = "[END_ENT]"
        context_tok = context.split()
        start_id = context_tok.index(start_str)
        end_id = context_tok.index(end_str)
        return " ".join(
            context_tok[
                max(start_id - self.truncate_length, 0) : end_id + self.truncate_length
            ]
        )
