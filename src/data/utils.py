###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
import os

# Note:
# 1. The current data release does not include the chain-of-thought reasoning datasets, which are commented out below.
# 2. We start from the "HuggingFaceH4/oasst1_en" dataset processed by HuggingFaceH4. We further processed the dataset such that each entry consists of one turn of dialogue.
# 3. We use "Zaid/coqa_expanded" processed by Zaid.
dataset_ids_by_split = {
    "lm_train": [
        "XSpace2000/oasst1_en",
        "commonsense_qa",
        "math_qa",
        "web_questions",
        "wiki_qa",
        "yahoo_answers_qa",
        "Zaid/coqa_expanded",
        "drop",
        "narrativeqa",
        "lucadiliello/newsqa",
        "pubmed_qa",
        "quail",
        "quarel",
        "squad_v2",
        "cnn_dailymail",
        # "aqua_rat",
        # "ecqa",
        # "gsm8k",
        # "competition_math",
        # "strategyqa"
    ],
    "retriever_train": [
        "commonsense_qa",
        "math_qa",
        "web_questions",
        "wiki_qa",
        "yahoo_answers_qa",
        "freebase_qa",
        "ms_marco",
        "pubmed_qa",
        "quarel",
        "oasst1_en",
        "sampled_corpus",
    ],
}


dataset_version = {
    "pubmed_qa": "pqa_labeled",
    "cnn_dailymail": "1.0.0",
    "ms_marco": "v1.1",
}


def get_task_subdir(task_name: str):
    if task_name in dataset_version:
        version = dataset_version[task_name]
        return f"{task_name}__{version}"
    else:
        return task_name


def is_qa_task(task_name: str):
    return task_name in [
        "commonsense_qa",
        "drop",
        "math_qa",
        "narrativeqa",
        "newsqa",
        "pubmed_qa",
        "quail",
        "quarel",
        "squad_v2",
        "web_questions",
        "wiki_qa",
        "yahoo_answers_qa",
        "nq",
        "tqa",
        "eli5",
        "hotpotqa",
        "zsrequestion",
    ]


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


def get_dev_path(task_dir: str):
    """
    Get the file path of the validation split of a dataset.
    """
    for file_name in os.listdir(task_dir):
        if file_name in [
            "valid.jsonl",
            "validation.jsonl",
            "dev.jsonl",
        ] or file_name.endswith("-dev-kilt.jsonl"):
            path = os.path.join(task_dir, file_name)
            return path

    if "web_questions" in task_dir:
        file_name = "test.jsonl"
        path = os.path.join(task_dir, file_name)
        if os.path.exists(path):
            return path

    print(f"Warning: cannot find dev file in {task_dir}")
    return None
