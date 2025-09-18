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
from src.data.eval.utils import f1_score, normalize_answer

# Wizard of Wikipedia example
# {
#     "input": "I like to watch ice hockey on TV. My favorite team is the Chicago Blackhawks.",
#     "output": [{"answer": "The Blackhawks are one of my favorite teams, they've won 6 Stanley Cup Championships since they started in 1926"
#                 "provenance": [
#                       {"wikipedia_id": "73126",
#                        "title": "Chicago Blackhawks",
#                        "start_paragraph_id": 1,
#                        "start_character": 260,
#                        "end_paragraph_id": 1,
#                        "end_character": 333,
#                        "bleu_score": 1.0,
#                        "section": "Section::::Abstract."
#                       }
#                  ]}]
# }


class WizardOfWikipediaTask(GenerationTask):
    dataset_dirs = ["wow"]
    metrics = [
        "f1",
        "nll",
        "nll_char",
        "nll_token",
        "oracle_f1",
        "nll_char_f1",
        "nll_token_f1",
    ]
    fewshot_mode = "first"
    fewshot_index: List[int] = [12608, 5394, 664, 21487, 16153]  # random
    n_fewshot = 0
    sep = "\n\n\n"
    target_prefix = "A:"
    fewshot_file = "wow-train-kilt-20000.jsonl"
    eval_file = "wow-dev-kilt.jsonl"
    prompt_format = "{context}\n{target_prefix}"
    max_text_len = 1984
    max_gen_len = 64

    def textify_example(self, example: Example) -> str:
        context = self.get_prompt(example)
        answer = self.get_target(example)
        return f"{context} {answer}"

    def get_prompt(self, example: Example) -> str:
        prompt = self.prompt_format.format(
            context=add_prefix_to_lines(example["input"]),
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
            example["output"][i]["answer"] for i in range(len(example["output"]))
        ]
        sample_metrics = {
            "f1": 100 * f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics


def add_prefix_to_lines(lines):
    lines = lines.split("\n")
    prefixes = ["Q:", "A:"]
    if len(lines) % 2 == 0:
        prefixes = ["A:", "Q:"]

    outputs = []
    for line_num, line in enumerate(lines):
        clean_line = " ".join(line.split()).strip()
        proc_line = prefixes[line_num % 2] + " " + clean_line
        outputs.append(proc_line)
    return "\n".join(outputs)
