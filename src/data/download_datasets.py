###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""
Download experiment datasets using Huggingface datasets library.

We further processed the HuggingFaceH4/oasst1_en dataset such that each entry consists of one turn of dialogue.

Usage:
    python3 src/data/download_datasets.py --root_dump_dir /tmp/radit_datasets/raw
"""

import argparse
import json
import os
import sys

from datasets import load_dataset
from utils import dataset_ids_by_split, dataset_version


def process_oasst1(data_dir):
    user_tag = "Q:"
    assistant_tag = "A:"

    for split in os.listdir(data_dir):
        if split in ["train.jsonl", "test.jsonl"]:
            in_jsonl = os.path.join(data_dir, split)
            orig_jsonl = in_jsonl.replace(".jsonl", ".orig.jsonl")
            if os.path.exists(orig_jsonl):
                print(
                    f"skip processing since data file is already processed; original data file {orig_jsonl} "
                )
                continue
            with open(in_jsonl) as f:
                formatted_data = []
                for conv_id, line in enumerate(f):
                    example = json.loads(line)
                    messages = example["messages"]
                    if len(messages) == 0 or messages[0]["role"] == "assistant":
                        # discard dialogues that start with the assistant
                        continue
                    question, answer, history = "", "", ""
                    turn_id = 0
                    for _, message in enumerate(messages):
                        data_point = dict()
                        if message["role"] == "user":
                            question = message["content"]
                        elif message["role"] == "assistant":
                            answer = message["content"]
                            data_point = {
                                "source": "oasst1",
                                "history": history,
                                "question": question,
                                "answer": {"input_text": answer},
                                "turn_id": turn_id,
                                "conv_id": conv_id,
                            }
                            turn = f"{user_tag} {question}\n{assistant_tag} {answer}"
                            history = (
                                turn if len(history) == 0 else (history + "\n\n" + turn)
                            )
                            formatted_data.append(data_point)
                            turn_id += 1
            out_jsonl = in_jsonl.replace(".jsonl", ".tmp.jsonl")
            with open(out_jsonl, "w") as o_f:
                for example in formatted_data:
                    o_f.write(json.dumps(example) + "\n")

            # rename files
            os.rename(in_jsonl, orig_jsonl)
            os.rename(out_jsonl, in_jsonl)
            print(f"oasst1 {split} split renamed to {orig_jsonl}")
            print(f"processed oasst1 {split} split saved to {in_jsonl}")


def main(args):
    if not os.path.exists(args.root_dump_dir):
        print(
            f'Dump directory "{args.root_dump_dir}" does not exist. Create it manually.'
        )
        sys.exit(0)
    download_save_dir = os.path.join(args.root_dump_dir, "raw")
    if not os.path.exists(download_save_dir):
        print(f"Creating directory {download_save_dir} to save downloaded data")
        os.makedirs(download_save_dir)

    dataset_ids = set(
        [
            dataset_id
            for split in dataset_ids_by_split
            for dataset_id in dataset_ids_by_split[split]
        ]
    )
    for dataset_id in dataset_ids:
        print(f"Loading {dataset_id}...")
        if "/" in dataset_id:
            sub_dir = dataset_id.rsplit("/", 1)[1]
        else:
            sub_dir = dataset_id

        if dataset_id in dataset_version:
            version = dataset_version[dataset_id]
            dataset = load_dataset(dataset_id, version)
            sub_dir = f"{dataset_id}__{version}"
        else:
            dataset = load_dataset(dataset_id, trust_remote_code=True)

        output_dir = os.path.join(download_save_dir, sub_dir)
        for split, split_dataset in dataset.items():
            output_jsonl = os.path.join(output_dir, f"{split}.jsonl")
            split_dataset.to_json(output_jsonl, orient="records", lines=True)
            print(f"{split} split saved to {output_jsonl}")
        print()

    # process OASST1 dataset
    process_oasst1(os.path.join(download_save_dir, "oasst1_en"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dump_dir",
        type=str,
        help="Dump directory to save the downloaded datasets.",
    )
    args = parser.parse_args()
    main(args)
