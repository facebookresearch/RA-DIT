###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""
Auxiliary Script:

Downsample the validation set of RA-DIT evaluation tasks to a given maximum number of 
examples (default: 2500) and assign a uniform name to validation sets.

Usage:
    python3 scripts/radit/down_sample_validation_sets.py \
        --eval-dir /tmp/datasets/raw \
        --tasks nq,tqa
"""
import glob
import os
import random

import fire

random.seed(42)
import subprocess

from data.utils import get_dev_path


def main(
    eval_dir: str,
    tasks: list = None,
    max_num_examples: int = 2500,
    tmp_save_dir: str = None,
    sample_ids_dir: str = None,
):
    """
    Args:
        eval_dir: Directory where the evaluation data is saved.
        tasks: List of tasks to downsample from.
        max_num_examples: Maximum number of examples per task.
        tmp_save_dir: Directory used to temporarily hold the downsampled files (to solve disk space limitation).
        sample_ids_dir: If specified, use the sample ID files for each task in this directory. This is used to ensure
            different eval datasets use the same examples.
    """
    if tasks is None:
        tasks = [
            task_name
            for task_name in os.listdir(eval_dir)
            if os.path.isdir(os.path.join(eval_dir, task_name))
        ]
        print(tasks)
    elif isinstance(tasks, str):
        tasks = [tasks]

    for task in tasks:
        task_dir = os.path.join(eval_dir, task)

        # Do not perform downsampling on MMLU dev sets
        if task == "mmlu":
            for subject in os.listdir(task_dir):
                subj_dir = os.path.join(task_dir, subject)
                if os.path.isdir(subj_dir):
                    sample_norm_jsonl = os.path.join(
                        subj_dir, f"valid.{max_n_examples}.jsonl"
                    )
                    if not os.path.exists(sample_norm_jsonl):
                        dev_path = os.path.join(subj_dir, "val.jsonl")
                        subprocess.run(["ln", "-s", dev_path, sample_norm_jsonl])
                        print(f"{sample_norm_jsonl} -> {dev_path}")
            continue

        # find data file corresponding to the validation split
        dev_path = get_dev_path(task_dir)
        if dev_path is None:
            import pdb

            pdb.set_trace()
            continue

        max_n_examples = 400 if task == "eli5" else max_num_examples

        with open(dev_path) as f:
            examples = f.readlines()
            sample_jsonl = dev_path.replace(".jsonl", f".{max_n_examples}.jsonl")
            sample_ids_txt = dev_path.replace(".jsonl", f".{max_n_examples}.ids.txt")

            if tmp_save_dir is not None:
                sample_jsonl = sample_jsonl.replace(eval_dir, tmp_save_dir)
                os.makedirs(os.path.dirname(sample_jsonl), exist_ok=True)
                sample_ids_txt = sample_ids_txt.replace(eval_dir, tmp_save_dir)
            sample_norm_jsonl = os.path.join(task_dir, f"valid.{max_n_examples}.jsonl")
            sample_norm_ids_txt = os.path.join(
                task_dir, f"valid.{max_n_examples}.ids.txt"
            )

            if len(examples) <= max_n_examples:
                subprocess.run(["ln", "-s", dev_path, sample_norm_jsonl])
            else:
                total_num_examples = len(examples)
                if sample_ids_dir is not None:
                    cached_sample_ids_txt = ""
                    _task = task.replace("_lsr", "")
                    for x in glob.glob(
                        os.path.join(
                            sample_ids_dir, _task, f"*.{max_n_examples}.ids.txt"
                        )
                    ):
                        cached_sample_ids_txt = x
                    print(f"Loading cached sample ids from {cached_sample_ids_txt}")
                    if not os.path.exists(cached_sample_ids_txt):
                        print(os.path.join(sample_ids_dir, task))
                    assert os.path.exists(cached_sample_ids_txt)
                    with open(cached_sample_ids_txt) as f:
                        sample_ids = [int(x.strip()) for x in f.readlines()]
                else:
                    sample_ids = random.sample(
                        range(total_num_examples), max_n_examples
                    )
                samples = [examples[i] for i in sample_ids]
                if not os.path.exists(sample_jsonl):
                    with open(sample_jsonl, "w") as o_f:
                        for example in samples:
                            o_f.write(example)
                        print(f"{len(samples)} saved to {sample_jsonl}")
                    with open(sample_ids_txt, "w") as o_f:
                        for id in sample_ids:
                            o_f.write(f"{id}\n")
                        print(f"{len(sample_ids)} saved to {sample_ids_txt}")
                else:
                    print(f"{sample_jsonl} exists!")

                if sample_jsonl != sample_norm_jsonl and not os.path.exists(
                    sample_norm_jsonl
                ):
                    subprocess.run(["ln", "-s", sample_jsonl, sample_norm_jsonl])
                    print(f"{sample_norm_jsonl} -> {sample_jsonl}")
                if sample_norm_ids_txt != sample_ids_txt and not os.path.exists(
                    sample_norm_ids_txt
                ):
                    subprocess.run(["ln", "-s", sample_ids_txt, sample_norm_ids_txt])
                    print(f"{sample_norm_ids_txt} -> {sample_ids_txt}")
        print()


if __name__ == "__main__":
    fire.Fire(main)
