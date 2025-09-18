###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""
Generate the retriever fine-tuning data for saving LSR scores.

All tasks are converted to a multiple-choice task, where each option is a *gold* answer.
LLM inference is then run to save LSR scores for each retrieved chunk for retriever FT.
Each sample will have fields "query", "answers" (as choices),
and optionally "gold_pos_support" and "gold_neg_support" (if support docs are provided).

Usage:
    PYTHONPATH=. python3 src/data/retriever_ft/prepare_retriever_ft_datasets.py \
        --input-dir $DATA_ROOT/raw \
        --save-dir  $DATA_ROOT/retriever_ft/lsr_score_saving_input/ \
        --task-list retriever_train
"""
import copy
import fire 
import glob
import jsonlines
import os
import random
random.seed(42)
import re
from tqdm import tqdm

from src.data.utils import add_prefix_to_lines, dataset_ids_by_split


NUM2LETTER = {i: chr(i + ord('A')) for i in range(26)}


def is_complete_sentence(s):
    return bool(re.search(r'[.!?"]$', s))


def convert_to_retriever_ft_lsr_format(in_jsonl_path, out_jsonl_path, task_name):
    with (
        jsonlines.open(in_jsonl_path) as reader,
        jsonlines.open(out_jsonl_path, "w") as writer
    ):
        for example in tqdm(reader):
            res = copy.deepcopy(example)
            if task_name == "oasst1_en":
                history = example["history"]
                question = example["question"].strip()
                answer = example["answer"]["input_text"].strip()
                if history:
                    res["query"] = f"{history}\n\n{question}"
                else:
                    res["query"] = question
                res["answers"] = {"A": answer}
            elif task_name == "commonsense_qa":
                res["query"] = example["question"].strip()
                if example["answerKey"] not in example["choices"]["label"]:
                    # no label for test set
                    continue
                answer_idx = example["choices"]["label"].index(example["answerKey"])
                res["answers"] = {
                    "A": example["choices"]["text"][answer_idx]
                }
            elif task_name == "freebase_qa":
                question = example["RawQuestion"]
                answers = set(ans["AnswersName"][0][0] for ans in example["Parses"]["Answers"])
                assert len(answers) <= 26
                res["query"] = question
                res["answers"] = {
                    NUM2LETTER[i]: ans for i, ans in enumerate(answers)
                }
            elif task_name == "math_qa":
                res["query"] = f"{example['Problem']} {example['options']}"
                # In addition to the gold answer, include rationale as another answer
                # FIXME the options are poorly formated and difficult to parse, ignore them for now.
                res["answers"] = {
                    "A": example["Rationale"].replace("explanation : ", "").strip(),
                    # "B": _parse_math_qa_options(example["options"]),
                }
            elif task_name == "ms_marco":
                res["query"] = example["query"]
                assert len(example["answers"]) <= 26
                res["answers"] = {
                    NUM2LETTER[i]: ans for i, ans in enumerate(example["answers"])
                }
                pos_ctxs, neg_ctxs = [], []
                for selected, passage in zip(example["passages"]["is_selected"], example["passages"]["passage_text"]):
                    if selected:
                        pos_ctxs.append({
                            "id": "",
                            "text": passage,
                            "title": "",
                        })
                    else:
                        neg_ctxs.append({
                            "id": "",
                            "text": passage,
                            "title": "",
                        })
                res["gold_pos_support"] = pos_ctxs
                res["gold_neg_support"] = neg_ctxs
            elif task_name == "pubmed_qa":
                res["query"] = example["question"]
                res["answers"] = {
                    "A": example["final_decision"],
                    "B": example["long_answer"],
                }
                res["gold_pos_support"] = [
                    {
                        "id": "",
                        "text": ctx,
                        "title": "",
                    } for ctx in example["context"]["contexts"]
                ]
            elif task_name == "quarel":
                res["query"] = example["question"]
                # harder to extract text answer, just use the letter answer
                res["answers"] = {
                    "A": NUM2LETTER[example["answer_index"]]
                }
            elif task_name == "web_questions":
                res["query"] = example["question"]
                res["answers"] = {
                    NUM2LETTER[i]: ans for i, ans in enumerate(example["answers"])
                }
            elif task_name in ["wiki_qa", "yahoo_answers_qa"]:
                res["query"] = example["question"]
                res["answers"] = {
                    "A": example["answer"]
                }
            elif task_name == "sampled_corpus":
                words = example["text"].split()
                if len(words) < 64:
                    # remove short passages
                    continue
                res["query"] = ' '.join(words[:len(words)//2])
                res["answers"] = {
                    "A": ' '.join(words[len(words)//2:])
                }
                res["chunk_id"] = example["chunk_id"] if "chunk_id" in example else example["id"]
            else:
                raise NotImplementedError()

            if not res["answers"]:
                print(f"No answer is found in sample for {task_name}; sample disgarded.")
                continue

            writer.write(res)


def get_raw_task_dir(instruct_dir, task_name):
    task_dir_prefix = os.path.join(instruct_dir, task_name)
    results = glob.glob(task_dir_prefix + "*")
    assert len(results) == 1, f"Expecting a single folder with prefix {task_name}, but found {len(results)}!"
    return results[0]


def main(input_dir: str, save_dir: str, task_list: str="retriever_train", overwrite_data: bool=False):
    assert(task_list in dataset_ids_by_split)

    for task_name in dataset_ids_by_split[task_list]:
        raw_task_dir = get_raw_task_dir(input_dir, task_name)
        out_dir = os.path.join(save_dir, task_name)
        os.makedirs(out_dir, exist_ok=True)

        print(os.path.join(raw_task_dir, "*.jsonl"))
        for in_jsonl in glob.glob(os.path.join(raw_task_dir, "*.jsonl")):
            if in_jsonl.endswith(".orig.jsonl"):
                # exclude unprocessed splits in the original format
                continue
            file_name = os.path.basename(in_jsonl)
            print(f"processing {in_jsonl}")
            out_jsonl = os.path.join(out_dir, file_name)
            convert_to_retriever_ft_lsr_format(in_jsonl, out_jsonl, task_name)
            print(f"retriever tuning data saved to {out_jsonl}")


if __name__ == "__main__":
    fire.Fire(main)
