###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
import os
import string

import fire
import jsonlines
from tqdm.auto import tqdm


def transform_sample_for_lsr(sample, task_str):
    if task_str == "nq":
        answers = sample["answers"]
        sample["choices"] = {
            string.ascii_uppercase[i]: answers[i] for i in range(len(answers))
        }
        # all options are gold answers, set an arbitrary answer
        sample["answer"] = "A"
    elif task_str == "tqa":
        answers = sample["answers"]
        # TQA has too many answer variants
        # saving the "target" as the only choice to reduce inference cost
        sample["choices"] = {"A": sample["target"]}
    else:
        raise NotImplementedError(task_str)

    return sample


def main(input_jsonl_path, output_jsonl_path, task_str):
    """
    This script converts a GenerationTask into a ChoiceTask,
    in order to generate LM scores for each support doc for LSR retriever fine-tuning.
    """
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with jsonlines.open(input_jsonl_path) as reader, jsonlines.open(
        output_jsonl_path, mode="w"
    ) as writer:
        for item in tqdm(reader):
            writer.write(transform_sample_for_lsr(item, task_str))


if __name__ == "__main__":
    fire.Fire(main)
