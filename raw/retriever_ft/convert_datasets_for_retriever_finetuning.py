###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""
All tasks are converted to ChoiceTask, where each option is a *gold* answer.
The data will be consumed by LSRTask to save LSR scores for retriever FT.
Each sample will have fields "query", "answers" (as choices),
and optionally "gold_pos_support" and "gold_neg_support" (if support docs are provided).

Usage:
    PYTHONPATH=. python3 scripts/radit/convert_datasets_for_retriever_finetuning.py \
        --input-dir /tmp/radit_datasets/ft_raw \
        --save-dir /tmp/radit_datasets/ft \
        --task-list retriever_ft_final
"""

import copy
import glob
import json
import os
import random

import fire
import jsonlines

random.seed(42)
import re

from scripts.radit.utils import task_lists
from src.eval.task.wizard_of_wikipedia import add_prefix_to_lines
from tqdm import tqdm


NUM2LETTER = {i: chr(i + ord("A")) for i in range(26)}


def is_complete_sentence(s):
    return bool(re.search(r'[.!?"]$', s))


def convert_to_retriever_ft_lsr_format(in_jsonl_path, out_jsonl_path, task_name):
    with (
        jsonlines.open(in_jsonl_path) as reader,
        jsonlines.open(out_jsonl_path, "w") as writer,
    ):
        for example in tqdm(reader):
            res = copy.deepcopy(example)
            if task_name == "oasst1_en_expanded":
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
                res["answers"] = {"A": example["choices"]["text"][answer_idx]}
            elif task_name == "freebase_qa":
                question = example["RawQuestion"]
                answers = set(
                    ans["AnswersName"][0][0] for ans in example["Parses"]["Answers"]
                )
                assert len(answers) <= 26
                res["query"] = question
                res["answers"] = {NUM2LETTER[i]: ans for i, ans in enumerate(answers)}
            elif task_name == "math_qa":
                # This is very difficult to get right due to poorly formated dataset
                # Ignoring the choices for now
                # def _parse_math_qa_options(choice_str):
                #     choice_map = {}
                #     try:
                #         if choice_str.startswith("["):
                #             # FIXME unsafe code
                #             options = eval(choice_str)
                #         else:
                #             options = [s.strip() for s in choice_str.split(",")]
                #         for option in options:
                #             parts = option.split(')')
                #             if len(parts) != 2:
                #                 print(f"Ill-formed MathQA example: {choice_str}")
                #             label, text = parts[0], parts[-1]
                #             choice_map[label.strip()] = text.strip()
                #     except:
                #         breakpoint()
                #     return choice_map

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
                for selected, passage in zip(
                    example["passages"]["is_selected"],
                    example["passages"]["passage_text"],
                ):
                    if selected:
                        pos_ctxs.append(
                            {
                                "id": "",
                                "text": passage,
                                "title": "",
                            }
                        )
                    else:
                        neg_ctxs.append(
                            {
                                "id": "",
                                "text": passage,
                                "title": "",
                            }
                        )
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
                    }
                    for ctx in example["context"]["contexts"]
                ]
            elif task_name == "quarel":
                res["query"] = example["question"]
                # harder to extract text answer, just use the letter answer
                res["answers"] = {"A": NUM2LETTER[example["answer_index"]]}
            elif task_name == "web_questions":
                res["query"] = example["question"]
                res["answers"] = {
                    NUM2LETTER[i]: ans for i, ans in enumerate(example["answers"])
                }
            elif task_name in ["wiki_qa", "yahoo_answers_qa"]:
                res["query"] = example["question"]
                res["answers"] = {"A": example["answer"]}
            elif task_name == "sampled_corpus":
                words = example["text"].split()
                if len(words) < 64:
                    # remove short passages
                    continue
                res["query"] = " ".join(words[: len(words) // 2])
                res["answers"] = {"A": " ".join(words[len(words) // 2 :])}
                res["chunk_id"] = (
                    example["chunk_id"] if "chunk_id" in example else example["id"]
                )
            elif task_name == "nq":
                question = example["question"].strip()
                if not is_complete_sentence(question):
                    question = f"{question}?"
                res["query"] = question
                res["answers"] = {
                    NUM2LETTER[i]: ans for i, ans in enumerate(example["answers"])
                }
            elif task_name == "tqa":
                res["query"] = example["question"].strip()
                res["answers"] = {"A": example["target"].strip()}
            elif task_name == "eli5":
                res["query"] = example["input"].strip()
                # TODO: we're training with only one ELI5 answer
                assert len(example["output"]), "answer is required for LSR."
                res["answers"] = {"A": example["output"][0]["answer"].strip()}
            elif task_name in ["hotpotqa"]:
                res["query"] = example["input"].strip()
                res["answers"] = {
                    NUM2LETTER[i]: ans["answer"].strip()
                    for i, ans in enumerate(example["output"])
                }
            elif task_name == "zsrequestion":
                res["query"] = example["meta"]["template_questions"][0]
                res["answers"] = {
                    NUM2LETTER[i]: ans["answer"].strip()
                    for i, ans in enumerate(example["output"])
                }
            elif task_name == "wow":
                res["query"] = add_prefix_to_lines(example["input"]).strip()
                res["answers"] = {
                    NUM2LETTER[i]: ans["answer"].strip()
                    for i, ans in enumerate(example["output"])
                }
            elif task_name == "fever":
                res["query"] = example["input"].strip()
                answer_map = {
                    "SUPPORTS": "Yes",
                    "REFUTES": "No",
                    "NOT ENOUGH INFO": "Not Sure",
                }
                assert len(example["output"]), "answer is required for LSR"
                res["answers"] = {
                    NUM2LETTER[i]: answer_map[ans["answer"]]
                    for i, ans in enumerate(example["output"])
                }
            elif task_name == "conllyagotrunc":
                start_str = "[START_ENT]"
                end_str = "[END_ENT]"
                context_tok = example["input"].split()
                start_id = context_tok.index(start_str)
                end_id = context_tok.index(end_str)
                res["query"] = " ".join(context_tok[start_id + 1 : end_id])
                res["answers"] = {
                    NUM2LETTER[i]: ans["answer"].strip()
                    for i, ans in enumerate(example["output"])
                }
            elif task_name == "trex":
                res["query"] = example["input"].strip()
                answers = list(
                    dict.fromkeys(ans["answer"] for ans in example["output"]).keys()
                )
                if len(answers) >= 26:
                    print("too many gold answers!")
                    answers = [answers[0]]
                res["answers"] = {NUM2LETTER[i]: ans for i, ans in enumerate(answers)}
            else:
                raise NotImplementedError()

            if not res["answers"]:
                print(
                    f"No answer is found in sample for {task_name}; sample disgarded."
                )
                continue

            writer.write(res)


def get_raw_task_dir(instruct_dir, task_name):
    raw_task_dir = os.path.join(instruct_dir, task_name)
    assert os.path.exists(raw_task_dir), f"{raw_task_dir} does not exist!"
    return raw_task_dir


def main(
    input_dir: str,
    save_dir: str,
    task_list: str = "retriever_ft_final",
    overwrite_data: bool = False,
):
    assert task_list in task_lists

    for task_name in task_lists[task_list]:
        raw_task_dir = get_raw_task_dir(input_dir, task_name)
        out_dir = os.path.join(save_dir, task_name)
        os.makedirs(out_dir, exist_ok=True)

        print(os.path.join(raw_task_dir, "*.jsonl"))
        for in_jsonl in glob.glob(os.path.join(raw_task_dir, "*.jsonl")):
            file_name = os.path.basename(in_jsonl)
            # if file_name not in ["train.jsonl", "test.jsonl", "validation.jsonl", "dev.jsonl", "train_ift.jsonl", "test_ift.jsonl"]:
            #     print(f"skipping file: {in_jsonl}")
            #     import pdb
            #     pdb.set_trace()
            #     continue

            # split = file_name.replace(".jsonl", "")
            print(f"processing {in_jsonl}")
            out_jsonl = os.path.join(out_dir, file_name)
            convert_to_retriever_ft_lsr_format(in_jsonl, out_jsonl, task_name)
            print(f"retriever tuning data saved to {out_jsonl}")


if __name__ == "__main__":
    fire.Fire(main)
