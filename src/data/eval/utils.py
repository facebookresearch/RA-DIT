###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
import logging
import os
import re
import string
import subprocess
from collections import Counter
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional

from numpy.random import RandomState

logger = logging.getLogger()
try:
    from rouge import Rouge
except ImportError:
    logger.warning("Please install rouge: pip install rouge")

"""
Normalization and score functions from SQuAD evaluation script
https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
"""


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(s: str) -> str:
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em(prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def alr(prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]):
    return float(normalize_fn(ground_truth) in normalize_fn(prediction))


def f1(prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score(
    prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]
):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])


def exact_match_score(
    prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]
):
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])


def answer_level_recall(
    prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]
):
    return max([alr(prediction, gt, normalize_fn) for gt in ground_truths])


# ROUGEL score definition
# source: https://github.com/facebookresearch/KILT/blob/main/kilt/eval_downstream.py
def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_score(prediction, ground_truths):
    return max(
        [_rougel_score(prediction, ground_truth) for ground_truth in ground_truths]
    )


def get_task_rng(
    base_seed: int = 42,
    different_seed_for_mp_groups: bool = False,
    different_seed_for_tasks_in_job_array: bool = False,
) -> RandomState:
    """
    Please use this RandomState for creating the data in your eval tasks
    Base seed needs to be the same within a model parallel group

    different_seed_for_mp_groups: bool -> If you want different GPUs to
        have different seed, please set it to True (it will make sure that it is
        the same within the same mp group, but different between mp groups)
    different_seed_for_tasks_in_job_array -> If you want the seed to be different
       between tasks in job array (to have different prompts sampled for majority
       voting for instance)

    """
    seed = [base_seed]
    if different_seed_for_tasks_in_job_array:
        # will give different seed for jobs arrays
        seed.append(different_seed_when_job_array(0))

    logger.info(f"Creating rng with seed: {tuple(seed)}")

    return RandomState(tuple(seed))


def different_seed_when_job_array(seed: int) -> int:
    # when using array jobs, we want different seed for each job
    return seed + int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


def is_valid_code(
    code: str,
    test: str,
    timeout: Optional[float] = 1.0,
):
    with TemporaryDirectory() as tmpdirname:
        actual_code = "\n".join([l for l in code.split("\n")])
        actual_test = "\n".join([l for l in test.split("\n")])
        script = f"""{actual_code}

{actual_test}
"""
        with open(os.path.join(tmpdirname, "script.py"), "w") as f:
            f.write(script)

        cmd = [
            "firejail",
            "--cpu=0",
            f"--rlimit-as={1024**3}",
            "python3",
            os.path.join(tmpdirname, "script.py"),
        ]
        try:
            run = subprocess.run(cmd, timeout=timeout, check=True, capture_output=True)
        except subprocess.TimeoutExpired as e:
            return False
        except subprocess.CalledProcessError as e:
            return False
        return run.returncode == 0
