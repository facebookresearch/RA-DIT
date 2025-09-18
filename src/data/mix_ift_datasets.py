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
    PYTHONPATH=. python3 scripts/radit/mix_ift_datasets.py \
        --instruct-dir /checkpoint/${USER}/retrofit/datasets/instruct/ \
        --save-dir /fsx-labs/${USER}/data/tengu/shuffled/ra-dit.it/
"""
import collections
import copy
import fire
import functools
import numpy as np
import os
import random
random.seed(42)
from typing import List

from scripts.radit.utils import task_lists, get_dev_path


tasks_with_passages = [
    "aqua",
    "commonsense_qa",
    "ecqa",
    "gsm8k",
    "math",
    "math_qa",
    "oasst1_en_expanded",
    "pubmed_qa",
    "quarel",
    "strategyqa",
    "web_questions",
    "wiki_qa",
    "yahoo_answers_qa"
]

tasks_with_passages_and_gt = [
    "pubmed_qa",
]


def safe_link(src, tgt):
    if (os.path.exists(tgt) and not os.path.islink(tgt)) or \
            (os.path.islink(tgt) and os.path.readlink(tgt) != src):
        os.remove(tgt)
    if not os.path.exists(tgt):
        os.symlink(src, tgt)


def upsample_without_repetition(l, r):
    r_int = int(np.floor(r))
    r_frac = r - r_int

    new_l = l * r_int
    partial_size = int(r_frac * len(l))
    new_l += l[:partial_size]
    return new_l


def mix_dataset_split(task_list: List[str], split: str, instruct_dir:str, data_version: str="v1", max_num_examples: int=-1):
    split_examples = collections.defaultdict(list)

    instruct_dir = instruct_dir.rstrip("/")
    instruct_dirname = os.path.basename(instruct_dir)
    n_support = int(instruct_dir.split(".")[2]) if "w_passages" in instruct_dirname else 0

    for task_name in task_list:
        task_dir = os.path.join(instruct_dir, task_name)
        split_jsonl = os.path.join(task_dir, f"{split}.inst.jsonl") if data_version == "v0" else \
                os.path.join(task_dir, f"{split}.jsonl")
        
        if split == "validation":
            split_jsonl = get_dev_path(task_dir) 
            if split_jsonl is None:
                print(f"Skipping dev set processing of {task_name}")
                continue
            if data_version == "v0":
                split_jsonl.replace(".jsonl", ".inst.jsonl")
        
        if not os.path.exists(split_jsonl):
            print(f"Warning: {split_jsonl} does not exist, skip saving")
            continue

        with open(split_jsonl) as f:
            split_examples[task_name] = f.readlines()
            random.shuffle(split_examples[task_name])
        
    if max_num_examples < 0:
        mixed_split_examples = [example for task_name in task_list for example in split_examples[task_name]]
    else:
        task_sizes = np.array([len(split_examples[task_name]) for task_name in task_list])
        _task_sizes = copy.deepcopy(task_sizes)
        if n_support > 0:
            # adjust task sizes
            for i, task_name in enumerate(task_list):
                if task_name in tasks_with_passages:
                    if task_name in tasks_with_passages_and_gt:
                        task_sizes[i] /= (n_support + 1)
                    else:
                        task_sizes[i] /= n_support
        task_sizes_trunc = np.array([task_sizes[i] if task_sizes[i] < max_num_examples else max_num_examples for i, _ in enumerate(task_list)])
        print("> task sizes after truncation:", "; ".join(f"{task_list[i]}: {task_sizes_trunc[i]}" for i in range(len(task_list))))
        print()

        task_dist_trunc = task_sizes_trunc / task_sizes_trunc.sum()
        print("> task distribution after truncation:", "; ".join(f"{task_list[i]}: {task_dist_trunc[i]:.3f}" for i in range(len(task_list))))
        print()

        total_num_examples = task_sizes.sum()
        print(f"total number of examples: {total_num_examples}")
        task_sizes_post_sampling = np.round(task_dist_trunc * total_num_examples).astype(int)
        print("> task sizes post sampling:", "; ".join(f"{task_list[i]}: {task_sizes_post_sampling[i]}" for i in range(len(task_list))))

        task_sample_ratio = task_sizes_post_sampling / _task_sizes
        print("> task sample ratio:", "; ".join(f"{task_list[i]}: {task_sample_ratio[i]:.3f}" for i in range(len(task_list))))

        # avoid throwing away any training samples by only performing upsampling while keeping the task ratio 
        task_sample_ratio /= task_sample_ratio.min()
        print("> task upsampling ratio:", "; ".join(f"{task_list[i]}: {task_sample_ratio[i]:.3f}" for i in range(len(task_list))))
        print()

        upsampled_split_examples = dict()
        for i, task_name in enumerate(task_list):
            upsampled_split_examples[task_name] = upsample_without_repetition(split_examples[task_name], task_sample_ratio[i])
        print("> task sizes post upsampling:", "; ".join(f"{task_name}: {len(upsampled_split_examples[task_name])}" for task_name in task_list))
        print()

        mixed_split_examples = [example for task_name in task_list for example in upsampled_split_examples[task_name]]

    print(f"Shuffling {len(mixed_split_examples)} examples in total")
    random.shuffle(mixed_split_examples)

    return mixed_split_examples


def main(instruct_dir: str, save_dir: str, task_list: str="p0", max_train_num_examples: int=7500):
    os.makedirs(save_dir, exist_ok=True)

    for split in ["train", "validation"]:
        print(f"Process {split} split")
        max_num_examples = max_train_num_examples if split == "train" else -1
        print(f"Setting up OASST1 dataset separately")
        _task_list = [t for t in task_lists[task_list] if "oasst1" not in t]

        mixed_split_examples = mix_dataset_split(
            task_list = _task_list, 
            split=split, 
            instruct_dir=instruct_dir, 
            data_version=task_list,
            max_num_examples=max_num_examples
        )
        if split == "train":
            out_jsonl = os.path.join(save_dir, "tasks", "it.chunk.00.jsonl")
        elif split == "validation":
            out_jsonl = os.path.join(save_dir, "tasks", "eval.jsonl")
        else:
            raise NotImplementedError

        with open(out_jsonl, 'w') as o_f:
            for example in mixed_split_examples:
                o_f.write(example)
        print(f"{len(mixed_split_examples)} saved to {out_jsonl}")
        print()

        if split == "train":
            # OASST1
            src_jsonl = os.path.join(instruct_dir, f"oasst1_en_expanded/{split}.jsonl")
            oasst1_save_dir = os.path.join(save_dir, "oasst1_en_expanded")
            os.makedirs(oasst1_save_dir, exist_ok=True)
            tgt_jsonl = os.path.join(oasst1_save_dir, "it.chunk.00.jsonl")
            safe_link(src_jsonl, tgt_jsonl):
            print(f"OASST1: {tgt_jsonl} created")
        elif split == "validation":
            tgt_jsonl = os.path.join(save_dir, "eval.jsonl")
            safe_link(out_jsonl, tgt_jsonl)
            print(f"Symlink created: {tgt_jsonl} -> {out_jsonl}")

    


if __name__ == "__main__":
    fire.Fire(main)
