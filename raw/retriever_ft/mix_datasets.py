###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""
Combine instruction tuning prompts of different datasets into a single file.

Usage:
    PYTHONPATH=. python3 scripts/radit/retriever_ft/mix_datasets.py \
        --input-dir /tmp/radit_datasets/instruct/ \
        --output-dir /tmp/radit_datasetss/ra-dit.it/ \
"""

import collections
import functools
import os
import random

import fire
import numpy as np

random.seed(42)
from typing import List

from scripts.radit.utils import get_dev_path, task_lists


def upsample_without_repetition(l, r):
    r_int = int(np.floor(r))
    r_frac = r - r_int

    new_l = l * r_int
    partial_size = int(r_frac * len(l))
    new_l += l[:partial_size]
    return new_l


def mix_dataset_split(
    task_list: List[str], split: str, instruct_dir: str, max_num_examples: int = -1
):
    split_examples = collections.defaultdict(list)
    for task_name in task_list:
        task_dir = os.path.join(instruct_dir, task_name)
        split_jsonl = os.path.join(task_dir, f"{split}.jsonl")
        if split == "validation":
            # use test set if validation set is not present
            if not os.path.exists(split_jsonl):
                split_jsonl = os.path.join(task_dir, "test.jsonl")

        if not os.path.exists(split_jsonl):
            print(f"Warning: {split_jsonl} does not exist, skip saving")
            continue

        with open(split_jsonl) as f:
            split_examples[task_name] = f.readlines()
            random.shuffle(split_examples[task_name])

    if max_num_examples < 0:
        mixed_split_examples = [
            example for task_name in task_list for example in split_examples[task_name]
        ]
    else:
        task_sizes = np.array(
            [len(split_examples[task_name]) for task_name in task_list]
        )
        task_sizes_trunc = np.array(
            [
                (
                    len(split_examples[task_name])
                    if len(split_examples[task_name]) < max_num_examples
                    else max_num_examples
                )
                for task_name in split_examples
            ]
        )
        print(
            "> task sizes after truncation:",
            "; ".join(
                f"{task_list[i]}: {task_sizes_trunc[i]}" for i in range(len(task_list))
            ),
        )

        task_dist_trunc = task_sizes_trunc / task_sizes_trunc.sum()
        print(
            "> task distribution after truncation:",
            "; ".join(
                f"{task_list[i]}: {task_dist_trunc[i]:.3f}"
                for i in range(len(task_list))
            ),
        )

        total_num_examples = task_sizes.sum()
        task_sizes_post_sampling = np.round(
            task_dist_trunc * total_num_examples
        ).astype(int)
        task_sample_ratio = task_sizes_post_sampling / task_sizes
        # avoid throwing away any training samples by only performing upsampling while keeping the task ratio
        task_sample_ratio /= task_sample_ratio.min()
        print(
            "> task upsampling ratio:",
            "; ".join(
                f"{task_list[i]}: {task_sample_ratio[i]:.3f}"
                for i in range(len(task_list))
            ),
        )

        upsampled_split_examples = dict()
        for i, task_name in enumerate(task_list):
            upsampled_split_examples[task_name] = upsample_without_repetition(
                split_examples[task_name], task_sample_ratio[i]
            )
        print(
            "> task sizes post upsampling:",
            "; ".join(
                f"{task_name}: {len(upsampled_split_examples[task_name])}"
                for task_name in task_list
            ),
        )

        mixed_split_examples = [
            example
            for task_name in task_list
            for example in upsampled_split_examples[task_name]
        ]

    print(f"Shuffling {len(mixed_split_examples)} examples in total")
    random.shuffle(mixed_split_examples)

    return mixed_split_examples


def main(
    input_dir: str,
    output_dir: str,
    task_list: str = "retriever_ft_final",
    max_train_num_examples: int = -1,
):
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "validation"]:
        print(f"Process {split} split")
        max_num_examples = max_train_num_examples if split == "train" else -1
        mixed_split_examples = mix_dataset_split(
            task_lists[task_list], split, input_dir, max_num_examples
        )
        if split == "train":
            out_jsonl = os.path.join(output_dir, "train.jsonl")
        else:
            out_jsonl = os.path.join(output_dir, "val.jsonl")
        with open(out_jsonl, "w") as o_f:
            for example in mixed_split_examples:
                o_f.write(example)
        print(f"{len(mixed_split_examples)} saved to {out_jsonl}")
        print()


if __name__ == "__main__":
    fire.Fire(main)
