###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""
Sample 64 examples from eval datasets and convert them into instruction tuning format.

Usage:
    PYTHONPATH=. python3 src/data/eval/prepare_64_shot_eval_examples.py \
        --eval-dir /tmp/radit_datasets/eval/replug_eval.cc.wiki.390m/ \
        --save-dir /tmp/radit_datasets/raw/eval_64shot/
"""

import copy
import glob
import json
import os
import random

import fire

random.seed(42)

import src.eval.task as task
from src.data.eval.task.conllyago import CoNLLYAGOTruncTask
from src.data.eval.task.eli5 import ELI5Task
from src.data.eval.task.fever import FEVERTask
from src.data.eval.task.hotpotqa import HotpotQATask
from src.data.eval.task.nq import NQTask
from src.data.eval.task.tqa import TQATask
from src.data.eval.task.trex import TRExTask
from src.data.eval.task.wizard_of_wikipedia import WizardOfWikipediaTask
from src.data.eval.task.zeroshot_re import ZeroShotRETask
from src.data.lm_ft.prepare_lm_ft_datasets import convert_to_rait_prompts


class_names = {
    "conllyagotrunc": CoNLLYAGOTruncTask,
    "eli5": ELI5Task,
    "fever": FEVERTask,
    "hotpotqa": HotpotQATask,
    "nq": NQTask,
    "tqa": TQATask,
    "trex": TRExTask,
    "wow": WizardOfWikipediaTask,
    "zsrequestion": ZeroShotRETask,
}


# random 64 examples used with Atlas
example_ids_64shots = {
    "nq": [
        27144,
        14489,
        49702,
        38094,
        6988,
        60660,
        65643,
        48249,
        48085,
        52629,
        48431,
        7262,
        34659,
        24332,
        44839,
        17721,
        50819,
        62279,
        37021,
        77405,
        52556,
        23802,
        40974,
        64678,
        69673,
        77277,
        18419,
        25635,
        1513,
        11930,
        5542,
        13453,
        52754,
        65663,
        67400,
        42409,
        74541,
        33159,
        65445,
        28572,
        74069,
        7162,
        19204,
        63509,
        12244,
        48532,
        72778,
        37507,
        70300,
        29927,
        18186,
        27579,
        58411,
        63559,
        4347,
        59383,
        57392,
        42014,
        77920,
        45592,
        32321,
        3422,
        61041,
        34051,
    ],
    "tqa": [
        75927,
        38807,
        452,
        68095,
        44621,
        34592,
        36091,
        65286,
        56484,
        48197,
        34692,
        28011,
        16670,
        62641,
        37865,
        6658,
        45724,
        37527,
        17740,
        31133,
        8010,
        48573,
        53670,
        15514,
        25996,
        54404,
        10739,
        55105,
        66122,
        73324,
        41202,
        71253,
        41258,
        51344,
        60092,
        50455,
        65078,
        36169,
        33408,
        55106,
        40526,
        65582,
        66337,
        39766,
        77174,
        17289,
        7367,
        50930,
        21151,
        21809,
        52804,
        26110,
        54414,
        73358,
        11459,
        66019,
        41084,
        13349,
        39059,
        6626,
        25540,
        15110,
        53320,
        61313,
    ],
}


def get_train_file_path(eval_dir, task_name):
    if task_name in ["nq", "tqa"]:
        return os.path.join(eval_dir, task_name, "train.jsonl")
    else:
        # KILT tasks
        if task_name == "conllyagotrunc":
            return os.path.join(eval_dir, task_name, "aidayago2-train-kilt.jsonl")
        train_file = None
        for train_file in glob.glob(
            os.path.join(eval_dir, task_name, "*-train-kilt-20000.jsonl")
        ):
            break
        assert train_file is not None
        return train_file


def get_train_sample_ids(task_name):
    if task_name in ["nq", "tqa"]:
        return example_ids_64shots[task_name]
    elif task_name == "conllyagotrunc":
        example_ids = copy.deepcopy(class_names[task_name].fewshot_index)
        _example_ids = list(range(18395))
        random.shuffle(_example_ids)
        for idx in _example_ids:
            if idx not in example_ids:
                example_ids.append(idx)
            if len(example_ids) == 64:
                break
        return example_ids
    else:
        # take the first 64 examples from the subsampled train files of KILT tasks
        return range(64)


def load_examples(train_jsonl):
    examples = []
    with open(train_jsonl) as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
    return examples


def main(eval_dir: str, save_dir: str, ignore_retrieved_passages: bool = False):
    eval_dir = eval_dir.rstrip("/")
    save_dir = save_dir.rstrip("/")
    all_examples = []
    for task_name in class_names:
        print(f"processing {task_name}")
        train_jsonl = get_train_file_path(eval_dir, task_name)
        sample_ids = get_train_sample_ids(task_name)
        examples = load_examples(train_jsonl)
        for sample_id in sample_ids:
            example = copy.deepcopy(examples[sample_id])
            example["task"] = task_name
            assert "sample_id" not in example
            example["sample_id"] = sample_id
            all_examples.append(example)

    assert os.path.exists(os.path.dirname(save_dir))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    out_jsonl = os.path.join(save_dir, "train.jsonl")
    with open(out_jsonl, "w") as o_f:
        for example in all_examples:
            o_f.write(f"{json.dumps(example)}\n")
    print(f"{len(all_examples)} saved to {out_jsonl}")

    # convert to RAIT format
    out_jsonl = os.path.join(save_dir, "it.chunk.00.jsonl")
    with open(out_jsonl, "w") as o_f:
        for example in all_examples:
            task_name = example["task"]
            for inst_example, _ in convert_to_rait_prompts(
                example,
                task_name,
                "train",
                ignore_retrieved_passages=ignore_retrieved_passages,
            ):
                o_f.write(f"{json.dumps(inst_example)}\n")
    print(f"{len(all_examples)} examples saved to {out_jsonl}")


if __name__ == "__main__":
    fire.Fire(main)
