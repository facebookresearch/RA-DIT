###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""
Sample 64 examples from eval datasets and add them to training.

Usage:
    PYTHONPATH=. python3 scripts/radit/gather_64_shot_eval_ft_examples.py \
        --eval-dir /tmp/radit_datasets/eval/replug_eval.cc.wiki.390m/ \
        --save-dir /tmp/radit_datasets/ft.64shot/eval_64shot/
"""

import copy
import glob
import json
import os
import random

import fire
import jsonlines

random.seed(42)
import src.eval.task as task
from src.eval.task.conllyago import CoNLLYAGOTruncTask
from src.eval.task.eli5 import ELI5Task
from src.eval.task.fever import FEVERTask
from src.eval.task.hotpotqa import HotpotQATask
from src.eval.task.nq import NQTask
from src.eval.task.tqa import TQATask
from src.eval.task.trex import TRExTask
from src.eval.task.wizard_of_wikipedia import WizardOfWikipediaTask
from src.eval.task.zeroshot_re import ZeroShotRETask
from tqdm import tqdm


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
        # eval_dir2 = os.path.join(os.path.dirname(eval_dir), "eval")
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


def main(eval_dir: str, save_dir: str):
    eval_dir = eval_dir.rstrip("/")
    for task_name in tqdm(class_names):
        print(f"processing {task_name}")
        train_jsonl = get_train_file_path(eval_dir, task_name)
        sample_ids = get_train_sample_ids(task_name)
        examples = load_examples(train_jsonl)
        selected = []
        for sample_id in sample_ids:
            example = examples[sample_id]
            example["task"] = task_name
            assert "sample_id" not in example
            example["sample_id"] = sample_id
            selected.append(example)
        task_save_dir = os.path.join(save_dir, task_name)
        os.makedirs(task_save_dir, exist_ok=True)
        with jsonlines.open(
            os.path.join(task_save_dir, "train.64shot.jsonl"), "w"
        ) as writer:
            for example in selected:
                writer.write(example)


if __name__ == "__main__":
    fire.Fire(main)
