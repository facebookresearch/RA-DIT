###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""
Generate the retrieval-augmented instruct tuning data for various datasets.

We assume all datasets for instruction tuning are saved under a root dir, 
e.g. /tmp/radit_datasets/raw

We perform an offline retrieval step which derives a retrieval query from each
example, and retrieve the relevant passages from a retrievel index. The 
retrieved passages are saved in the "retrieved_support" of the example JSON.

The formatted instruction tuning data is saved at the designated dump dir,
e.g. /tmp/radit_datasets/ft

Usage:
  
  1. Offline retrieval  
    PYTHONPATH=. python3 src/data/lm_ft/prepare_lm_ft_datasets.py \
        --input_dir /tmp/radit_datasets/raw \
        --save_dir /tmp/radit_datasets/raw.ra \
        --action retrieval

  2. Format examples
    i) Dump retrieval-augmented (RA) instruction-tuning (IT) data 
    PYTHONPATH=. python3 src/data/lm_ft/prepare_lm_ft_datasets.py \
        --input_dir /tmp/radit_datasets/raw \
        --save_dir /tmp/radit_datasets/ft.ra \
        --n_support 3 \
        --action format_data

    ii) Dump vanilla IT data without RA 
    PYTHONPATH=. python3 src/data/lm_ft/prepare_lm_ft_datasets.py \
        --input_dir /tmp/radit_datasets/raw \
        --save_dir /tmp/radit_datasets/ft \
        --action format_data \
        --ignore_retrieved_passages
"""

import argparse
import copy
import glob
import json
import os
import random

random.seed(42)
import re

from src.data.utils import (
    add_prefix_to_lines,
    dataset_ids_by_split,
    get_task_subdir,
    is_qa_task,
)
from tqdm import tqdm


# Instruction separators used by Llama
BEGIN_INST_TAG = "[INST]"
END_INST_TAG = "[/INST]"

NUM2LETTER = {i: chr(i + ord("A")) for i in range(26)}


def is_complete_sentence(s):
    return bool(re.search(r'[.!?"]$', s))


ext_kn_start_marker = "Background: "  # marker indicating the beginning of content representing external knowledge
ext_kn_end_marker = (
    "\n\n"  # marker indicating the end of content representing external knowledge
)


def convert_to_rait_prompts(
    example, task_name, split, n_support=3, ignore_retrieved_passages=False
):
    inst_example = dict()
    has_no_answer = False
    has_multi_answers = False

    question_start_marker = random.choice(["Q: ", "Question: ", ""])
    question_end_marker = "\n"
    answer_marker = random.choice(["A: ", "Answer: "])

    answers = None
    ext_knowledge = None

    if is_qa_task(task_name):
        # QA tasks
        choice_sep = "\n"
        choice_str = None

        # Note:
        #   1. For questions with multiple answers, we select only the first answer for training.
        #   2. We map all choice markers to A, B, C, D...
        if task_name == "commonsense_qa":
            question = example["question"].strip()
            choice_str = choice_sep.join(
                [
                    f"{l}. {t.strip()}"
                    for l, t in zip(
                        example["choices"]["label"], example["choices"]["text"]
                    )
                ]
            )
            answer = example["answerKey"].strip()
        elif task_name == "drop":
            ext_knowledge = example["passage"].strip()
            question = example["question"].strip()
            answers = example["answers_spans"]["spans"]
            answer = answers[0].strip()
        elif task_name == "math_qa":
            question = example["Problem"].strip()
            choice_str = example["options"]
            answer = example["Rationale"].replace("explanation : ", "").strip()
        elif task_name == "narrativeqa":
            ext_knowledge = example["document"]["summary"]["text"].strip()
            question = example["question"]["text"].strip()
            answers = example["answers"]
            answer = answers[0]["text"].strip()
        elif task_name == "newsqa":
            ext_knowledge = example["context"].strip()
            question = example["question"].strip()
            answers = example["answers"]
            answer = answers[0].strip()
        elif task_name == "pubmed_qa":
            ext_knowledge = "\n".join(
                [x.strip() for x in example["context"]["contexts"]]
            )
            question = example["question"].strip()
            answer = example["long_answer"].strip()
        elif task_name == "quail":
            ext_knowledge = example["context"].strip()
            question = example["question"].strip()
            choices = example["answers"]
            choice_str = choice_sep.join(
                [f"{NUM2LETTER[l]}. {t.strip()}" for l, t in enumerate(choices)]
            )
            answer = NUM2LETTER[example["correct_answer_id"]]
        elif task_name == "quarel":
            question = example["question"].strip()
            answer = NUM2LETTER[example["answer_index"]]
        elif task_name == "squad_v2":
            ext_knowledge = example["context"].strip()
            question = example["question"].strip()
            answers = example["answers"]["text"]
            if len(answers) == 0:
                has_no_answer = True
                answer = "I don't know."
            else:
                if len(answers) > 1:
                    has_multi_answers = True
                answer = answers[0].strip()
        elif task_name == "web_questions":
            question = example["question"].strip()
            answers = example["answers"]
            answer = "\n".join(answers).strip()
        elif task_name == "wiki_qa":
            question = example["question"].strip()
            answer = example["answer"].strip()
        elif task_name == "yahoo_answers_qa":
            question = example["question"].strip()
            answer = example["answer"].strip()
        elif task_name == "nq":
            question = example["question"].strip()
            if not is_complete_sentence(question):
                question = f"{question}?"
            # TODO: we're training with only one NQ answer
            answer = example["answers"][0].strip()
        elif task_name == "tqa":
            question = example["question"].strip()
            answer = example["target"].strip()
        elif task_name == "eli5":
            question = example["input"].strip()
            # TODO: we're training with only one ELI5 answer
            if len(example["output"]) == 0:
                assert split == "test"
            answer = (
                example["output"][0]["answer"].strip()
                if len(example["output"]) > 0
                else ""
            )
        elif task_name in ["hotpotqa"]:
            question = example["input"].strip()
            answers = example["output"]
            answer = answers[0]["answer"].strip()
        elif task_name == "zsrequestion":
            question = example["meta"]["template_questions"][0]
            answers = example["output"]
            answer = answers[0]["answer"].strip()

        if choice_str is not None:
            question = f"{question}\n{choice_str}"
        inst_example["text"] = (
            f"{question_start_marker}{question}{question_end_marker}{answer_marker}{END_INST_TAG} {answer}"
        )

    elif task_name in ["oasst1_en", "coqa_expanded", "wow"]:
        # Dialgoue tasks
        if task_name == "coqa_expanded":
            ext_knowledge = example["story"]
            question = example["question"].strip()
            answer = example["answer"]["input_text"].strip()
            inst_example["text"] = f"Q: {question}\nA: {END_INST_TAG} {answer}"
        elif task_name == "oasst1_en":
            history = example["history"].strip()
            question = example["question"].strip()
            answer = example["answer"]["input_text"].strip()
            if len(history) == 0:
                inst_example["text"] = (
                    f"{question_start_marker}{question}{question_end_marker}{answer_marker}{END_INST_TAG} {answer}"
                )
            else:
                inst_example["text"] = (
                    f"{history}\n\nQ: {question}\nA: {END_INST_TAG} {answer}"
                )
        elif task_name == "wow":
            context = add_prefix_to_lines(example["input"]).strip()
            answers = example["output"]
            answer = answers[0]["answer"].strip()
            inst_example["text"] = f"{context}\nA: {END_INST_TAG} {answer}"

    elif task_name in ["aqua", "ecqa", "gsm8k", "math", "strategyqa"]:
        # Reasoning tasks
        question = example["src"].strip()
        answer = example["tgt"].strip()
        inst_example["text"] = (
            f"{question_start_marker}{question} {END_INST_TAG} {answer}"
        )

    else:
        # MISC tasks
        if task_name == "cnn_dailymail":
            ext_knowledge = example["article"].strip()
            instruction = random.choice(
                [
                    "Can you write an outline of this article in a few points?",
                    "Write a summary of this article:",
                    "Summarize this article:",
                    "In 2 or 3 sentences, what are the main points one should remember from this news article?",
                    "Could you please generate a TLDR (Too Long Didn't Read) summary of this news article?",
                    "Condense the article down to the essentials to present it in the form of short cards in mobile news apps:",
                    "Summarize the following article in brief:",
                    "Extract key points from the article based on which the stock market could react:",
                    "Write highlights for this article:",
                    "Write some highlights for this article:",
                    "What are highlight points for this article?",
                    "Summarize the highlights of this article.",
                    "Here is a summary of the highlights for this article:",
                ]
            )  # from PromptSource and FLAN
            answer = example["highlights"].strip()
        elif task_name == "fever":
            claim = example["input"].strip()
            instruction = f"Is this statement true? {claim}"
            answer_map = {
                "SUPPORTS": "Yes",
                "REFUTES": "No",
                "NOT ENOUGH INFO": "Not Sure",
            }
            if len(example["output"]) == 0:
                assert split == "test"
            answer = (
                answer_map[example["output"][0]["answer"]]
                if len(example["output"]) > 0
                else ""
            )
        elif task_name == "conllyagotrunc":
            context = example["input"].strip()
            instruction = f"{context}\nAssign a unique identity to entities (surrounded by [START_ENT] and [END_ENT]) mentioned in the text"
            answers = example["output"]
            answer = answers[0]["answer"].strip()
        elif task_name == "trex":
            instruction = example["input"].strip()
            answers = example["output"]
            answer = answers[0]["answer"].strip()

        inst_example["text"] = (
            f"{instruction}{question_end_marker}{answer_marker}{END_INST_TAG} {answer}"
        )

    if answers is not None:
        if len(answers) == 0:
            has_no_answer = True
        elif len(answers) > 1:
            has_multi_answers = True

    status = {
        "has_no_answer": has_no_answer,
        "has_multi_answers": has_multi_answers,
    }

    ext_knowledge_list = []
    # If original data release include background knowledge, add them to the external knowledge list.
    if ext_knowledge is not None:
        ext_knowledge_list.append(ext_knowledge)

    # Add retrieved passages to external knowledge list
    if not ignore_retrieved_passages:
        support_docs = example.get("retrieved_support", None)
        assert support_docs is not None
        for support_doc in support_docs[:n_support]:
            ext_knowledge_list.append(
                support_doc["title"].strip() + " " + support_doc["text"].strip()
            )

    # Generate examples
    inst_examples = list()

    if len(ext_knowledge_list) > 0:
        for ext_knowledge in ext_knowledge_list:
            ext_knowledge_str = (
                f"{ext_kn_start_marker}{ext_knowledge}{ext_kn_end_marker}"
            )

            # Ensure there is a separator between external knowledge and the task prompt
            if question_start_marker == "":
                question_start_marker = random.choice(["Q: ", "Question: "])
                ext_knowledge_str += question_start_marker
            _inst_example = copy.deepcopy(inst_example)
            _inst_example["text"] = (
                f"{BEGIN_INST_TAG} {ext_knowledge_str}" + _inst_example["text"]
            )
            inst_examples.append(_inst_example)
    else:
        if ignore_retrieved_passages:
            inst_examples = [inst_example]
        else:
            raise ValueError(
                f"{len(ext_knowledge_list)} external knowledge snippets for example {inst_example}. You may need to set '--ignore_retrieved_passages true'."
            )

    for _inst_example in inst_examples:
        # Add meta data information
        _inst_example["meta"] = {
            "dataset": task_name,
        }

        yield _inst_example, status


def main(args: argparse.Namespace):
    n_support = args.n_support

    save_dir = args.save_dir
    if not save_dir.endswith(f".n{n_support}"):
        save_dir = f"{save_dir}.n{n_support}"
    if n_support > 0 and args.ignore_retrieved_passages:
        save_dir = save_dir + ".no_retrieved"
    os.makedirs(save_dir, exist_ok=True)

    for dataset_id in dataset_ids_by_split["lm_train"]:
        task_name = dataset_id.split("/")[-1]
        task_subdir = get_task_subdir(task_name)
        raw_task_dir = os.path.join(args.input_dir, task_subdir)
        ft_task_dir = os.path.join(save_dir, task_subdir)
        assert os.path.exists(
            raw_task_dir
        ), f"Task data not found at {raw_task_dir}, check if path is correctly set."
        os.makedirs(ft_task_dir, exist_ok=True)
        print(f">>> processing task {task_name}")
        for file_name in os.listdir(raw_task_dir):
            if not file_name.endswith(".jsonl"):
                continue
            in_jsonl = os.path.join(raw_task_dir, file_name)
            print(f"processing {in_jsonl}")
            if file_name not in [
                "train.jsonl",
                "validation.jsonl",
                "dev.jsonl",
                "train_ift.jsonl",
            ]:
                if not file_name.endswith("orig.jsonl"):
                    print(
                        f"<!> skipping file (test split of the fine-tuning tasks) to save processing time: {in_jsonl}"
                    )
                continue

            out_jsonl = os.path.join(ft_task_dir, file_name)
            print(f"writing to {out_jsonl}")

            split = file_name.replace(".jsonl", "")
            with open(in_jsonl) as f, open(out_jsonl, "w") as o_f:
                num_examples = 0
                num_lines = 0
                num_no_answer = 0
                num_multi_answers = 0
                for line in tqdm(f):
                    example = json.loads(line)
                    if task_name == "wiki_qa" and example["label"] == 0:
                        continue
                    status = None
                    for processed_example, status in convert_to_rait_prompts(
                        example,
                        task_name,
                        split,
                        n_support,
                        args.ignore_retrieved_passages,
                    ):
                        o_f.write(json.dumps(processed_example) + "\n")
                        num_lines += 1
                    num_examples += 1

                    if status["has_no_answer"]:
                        num_no_answer += 1
                    if status["has_multi_answers"]:
                        num_multi_answers += 1

            print(
                f"instruction tuning prompts saved to {out_jsonl} ({num_examples} examples : {num_lines} lines)"
            )
            print(
                f"> {(num_no_answer / num_examples) * 100 :.1f}% ({num_no_answer}) examples do not have answers"
            )
            print(
                f"> {(num_multi_answers / num_examples) * 100 :.1f}% ({num_multi_answers}) examples have multiple answers"
            )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="directory where the fine-tuning data is saved (in the original Hugging Face dataset format)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="directory to save the retrieval-augmented data and formatted instruction tuning data",
    )
    parser.add_argument(
        "--n_support",
        type=int,
        default=3,
        help="use top n support documents to generate retrieval-augmented instruction tuning examples",
    )
    parser.add_argument(
        "--action",
        choices=("retrieval", "format_data"),
    )
    parser.add_argument(
        "--ignore_retrieved_passages",
        action="store_true",
        help="If set, do not use retrieved passages for data generation but still includes background information coming with the dataset release (e.g. for reading comprehension).",
    )
    parser.add_argument("--overwrite_data", action="store_true")

    args = parser.parse_args()
    main(args)
