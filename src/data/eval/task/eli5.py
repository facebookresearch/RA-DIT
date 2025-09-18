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
from src.data.eval.utils import rouge_score

# ELI5 example (KILT version)
# {
#     "id": "1kiwfx",
#     "input": "In Trading Places (1983, Akroyd/Murphy) how does the scheme at the end of the movie work? Why would buying a lot of OJ at a high price ruin the Duke Brothers?",
#     "output": [{"answer": "The final scene involves future contracts. This simply means entering into a contract to buy something (oil, wheat, even frozen concentrated orange juice(FCOJ)) at a specified time for ..",
#                 "provenance": [
#                       {"wikipedia_id": "242855",
#                        "title": "Futures contract",
#                        "start_paragraph_id": 1,
#                        "start_character": 14,
#                        "end_paragraph_id": 1,
#                        "end_character": 612,
#                        "bleu_score": 0.9232808519770748,
#                        "section": "Section::::Abstract."
#                       }
#                  ]}]
# }


class ELI5Task(GenerationTask):
    dataset_dirs = ["eli5"]
    metrics = [
        "nll",
        "nll_char",
        "nll_token",
        "oracle_rouge-l",
        "rouge-l",
        "nll_char_rouge-l",
        "nll_token_rouge-l",
    ]
    fewshot_mode = "first"
    # fewshot_index: List[int] = [13, 14, 74, 83, 106] # manual
    fewshot_index: List[int] = [152630, 33324, 41272, 233937, 201290]  # random
    n_fewshot = 5
    sep = "\n\n"
    target_prefix = "A:"
    fewshot_file = "eli5-train-kilt-20000-oracle.jsonl"
    eval_file = "eli5-dev-kilt.jsonl"
    prompt_format = "Q: {question}\n{target_prefix}"
    max_text_len = 1536
    max_gen_len = 512

    def textify_example(self, example: Example) -> str:
        context = self.get_prompt(example)
        answer = self.get_target(example)
        return f"{context} {answer}"

    def get_prompt(self, example: Example) -> str:
        prompt = self.prompt_format.format(
            question=example["input"],
            target_prefix=self.target_prefix,
        )
        return prompt

    def get_target(self, example: Example) -> str:
        answers = example["output"]
        target = answers[0]["answer"]
        return " ".join(target.split()).strip()

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
            "rouge-l": 100 * rouge_score(prediction, ground_truths),
        }
        return sample_metrics
