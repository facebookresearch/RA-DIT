###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
import json
import os
from pathlib import Path

import fire


def symlink(src, tgt):
    assert src.is_file()
    if tgt.is_file():
        if os.path.islink(tgt):
            tgt.unlink()
        else:
            os.remove(tgt)
    os.symlink(src, tgt)


def load_json(filepath):
    if filepath.suffix == ".json":
        return json.load(open(filepath))
    elif filepath.suffix == ".jsonl":
        return [json.loads(line) for line in open(filepath)]
    else:
        raise NotImplementedError(filepath)


def write_jsonl(data, filepath):
    with open(filepath, "w") as fp:
        for d in data:
            fp.write(json.dumps(d) + "\n")


def load_lst(filepath):
    return [line.strip() for line in open(filepath)]


def main(data_root_dir: str, tasks: list = None):
    """
    python scripts/eval/convert_datasets_for_commonsense.py /fsx-instruct-opt2/mingdachen/commonsense_eval/
    """
    data_root_dir = Path(data_root_dir)
    assert data_root_dir.is_dir()
    if isinstance(tasks, str):
        tasks = [tasks]
    if tasks is None:
        tasks = [
            "arc_all",
            "boolq",
            "hellaswag",
            "obqa",
            "piqa",
            "siqa",
            "winogrande_1.1",
        ]

    for task in tasks:
        if task == "arc_all":
            arc_challenge_src_dir = (
                data_root_dir / task / "raw_data" / "ARC-V1-Feb2018-2" / "ARC-Challenge"
            )
            arc_challenge_tgt_dir = data_root_dir / "arc_challenge"
            arc_challenge_tgt_dir.mkdir(exist_ok=True, parents=True)
            symlink(
                arc_challenge_src_dir / "ARC-Challenge-Train.jsonl",
                arc_challenge_tgt_dir / "train.jsonl",
            )
            symlink(
                arc_challenge_src_dir / "ARC-Challenge-Dev.jsonl",
                arc_challenge_tgt_dir / "val.jsonl",
            )
            symlink(
                arc_challenge_src_dir / "ARC-Challenge-Test.jsonl",
                arc_challenge_tgt_dir / "test.jsonl",
            )

            arc_easy_src_dir = (
                data_root_dir / task / "raw_data" / "ARC-V1-Feb2018-2" / "ARC-Easy"
            )
            arc_easy_tgt_dir = data_root_dir / "arc_easy"
            arc_easy_tgt_dir.mkdir(exist_ok=True, parents=True)
            symlink(
                arc_easy_src_dir / "ARC-Easy-Train.jsonl",
                arc_easy_tgt_dir / "train.jsonl",
            )
            symlink(
                arc_easy_src_dir / "ARC-Easy-Dev.jsonl", arc_easy_tgt_dir / "val.jsonl"
            )
            symlink(
                arc_easy_src_dir / "ARC-Easy-Test.jsonl",
                arc_easy_tgt_dir / "test.jsonl",
            )
        elif task == "boolq":
            boolq_src_dir = data_root_dir / task / "raw_data"
            boolq_tgt_dir = data_root_dir / task
            symlink(boolq_src_dir / "train.jsonl", boolq_tgt_dir / "train.jsonl")
            symlink(boolq_src_dir / "val.jsonl", boolq_tgt_dir / "val.jsonl")
            test_data = load_json(boolq_src_dir / "test.jsonl")
            for idx, tl in enumerate(test_data):
                test_data[idx]["label"] = True
            write_jsonl(test_data, boolq_tgt_dir / "test.jsonl")
        elif task == "hellaswag":
            hellaswag_src_dir = data_root_dir / task / "raw_data"
            hellaswag_tgt_dir = data_root_dir / task
            symlink(
                hellaswag_src_dir / "hellaswag_train.jsonl",
                hellaswag_tgt_dir / "train.jsonl",
            )
            symlink(
                hellaswag_src_dir / "hellaswag_val.jsonl",
                hellaswag_tgt_dir / "val.jsonl",
            )
            test_data = load_json(hellaswag_src_dir / "hellaswag_test.jsonl")
            for idx, tl in enumerate(test_data):
                test_data[idx]["label"] = 0
            write_jsonl(test_data, hellaswag_tgt_dir / "test.jsonl")
        elif task == "obqa":
            obqa_src_dir = (
                data_root_dir
                / task
                / "raw_data"
                / "OpenBookQA-V1-Sep2018"
                / "Data"
                / "Main"
            )
            obqa_tgt_dir = data_root_dir / task
            symlink(obqa_src_dir / "train.jsonl", obqa_tgt_dir / "train.jsonl")
            symlink(obqa_src_dir / "dev.jsonl", obqa_tgt_dir / "val.jsonl")
            symlink(obqa_src_dir / "test.jsonl", obqa_tgt_dir / "test.jsonl")
        elif task == "piqa":
            piqa_src_dir = data_root_dir / task / "raw_data"
            piqa_tgt_dir = data_root_dir / task
            train_data = load_json(piqa_src_dir / "train.jsonl")
            train_label = load_lst(piqa_src_dir / "train-labels.lst")
            assert len(train_data) == len(train_label)
            for idx, tl in enumerate(train_label):
                train_data[idx]["label"] = tl
            write_jsonl(train_data, piqa_tgt_dir / "train.jsonl")
            dev_data = load_json(piqa_src_dir / "valid.jsonl")
            dev_label = load_lst(piqa_src_dir / "valid-labels.lst")
            assert len(dev_data) == len(dev_label)
            for idx, tl in enumerate(dev_label):
                dev_data[idx]["label"] = tl
            write_jsonl(dev_data, piqa_tgt_dir / "val.jsonl")
            test_data = load_json(piqa_src_dir / "tests.jsonl")
            for idx, tl in enumerate(test_data):
                test_data[idx]["label"] = 0
            write_jsonl(test_data, piqa_tgt_dir / "test.jsonl")
        elif task == "siqa":
            siqa_src_dir = data_root_dir / task / "raw_data"
            siqa_tgt_dir = data_root_dir / task
            train_data = load_json(siqa_src_dir / "socialiqa-train-dev" / "train.jsonl")
            train_label = load_lst(
                siqa_src_dir / "socialiqa-train-dev" / "train-labels.lst"
            )
            assert len(train_data) == len(train_label)
            for idx, tl in enumerate(train_label):
                train_data[idx]["label"] = tl
            write_jsonl(train_data, siqa_tgt_dir / "train.jsonl")
            dev_data = load_json(siqa_src_dir / "socialiqa-train-dev" / "dev.jsonl")
            dev_label = load_lst(
                siqa_src_dir / "socialiqa-train-dev" / "dev-labels.lst"
            )
            assert len(dev_data) == len(dev_label)
            for idx, tl in enumerate(dev_label):
                dev_data[idx]["label"] = tl
            write_jsonl(dev_data, siqa_tgt_dir / "val.jsonl")
            test_data = load_json(siqa_src_dir / "socialiqa-test" / "socialiqa.jsonl")
            for idx, tl in enumerate(test_data):
                test_data[idx]["label"] = 0
            write_jsonl(test_data, siqa_tgt_dir / "test.jsonl")
        elif task == "winogrande_1.1":
            wino_src_dir = data_root_dir / task / "raw_data" / "winogrande_1.1"
            wino_tgt_dir = data_root_dir / task
            symlink(wino_src_dir / "train_xl.jsonl", wino_tgt_dir / "train.jsonl")
            symlink(wino_src_dir / "dev.jsonl", wino_tgt_dir / "val.jsonl")
            test_data = load_json(wino_src_dir / "test.jsonl")
            for idx, tl in enumerate(test_data):
                test_data[idx]["label"] = 0
            write_jsonl(test_data, wino_tgt_dir / "test.jsonl")
        else:
            raise NotImplementedError(task)


if __name__ == "__main__":
    fire.Fire(main)
