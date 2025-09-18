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
import string
from abc import abstractmethod
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F

from src.data.tokenizer import Tokenizer
from src.utils import mask_col_max

Example = Dict[str, Any]


class BaseTask:
    tasks: Dict[str, Type["BaseTask"]] = {}
    dataset_dirs: List[str] = []
    eval_mode: str
    metrics: List
    sep: str
    eval_file: str
    max_text_len: int
    random_fewshots: bool = False
    no_truncate_fewshot: bool = (
        False  # when enabled, it will always maintain complete few-shot examples
    )

    # options for fewshot mode: first, random and index
    # first: select the first n_fewshot examples of the fewshot file as fewshot examples for all test examples
    # random: select n_fewshot random examples from the fewshot file as fewshot examples, examples similar to the current test example will be discarded
    # index: select fewshot examples in the fewshot file at the index specific by fewshot_index list
    # for instance fewshot_index = [1, 4] will use the second and fifth examples as fewshot examples for all examples.
    fewshot_file: str
    fewshot_mode: str
    n_fewshot: int = 0
    fewshot_index: List[int] = []

    # options for REPLUG mode:
    use_ensemble_inference = False
    support_prefix = "Background:"
    support_sep: str = "\n\n"
    n_support: int = 3  # max number of support docs to use

    def __init__(self, tokenizer: Tokenizer, task_dir: str, **replug_args):
        self.tokenizer = tokenizer
        self.task_dir = task_dir
        self._all_fewshot_examples: List[Example] = []
        self.no_truncate_fewshot = replug_args.get("no_truncate_fewshot", False)

        n_fewshot = replug_args.get("n_fewshot", None)
        if n_fewshot is not None:
            self.n_fewshot = n_fewshot
            if n_fewshot == 0:
                self.fewshot_index = []

        self.use_ensemble_inference = replug_args.get("use_ensemble_inference", False)
        n_support = replug_args.get("n_support", None)
        if n_support is not None:
            self.n_support = n_support

        eval_file = replug_args.get("eval_file", None)
        if eval_file:
            self.eval_file = eval_file
        assert self.eval_file.endswith(".jsonl")

        if self.n_fewshot > 0 or len(self.fewshot_index) > 0:
            assert self.fewshot_mode in [
                "index",
                "first",
                "random",
            ], f"Fewshot strategy {self.fewshot_mode} not supported"
            assert os.path.isfile(
                os.path.join(self.task_dir, self.fewshot_file)
            ), f"Fewshot file does not exist {os.path.join(self.task_dir, self.fewshot_file)}"

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for dirname in cls.dataset_dirs:
            cls.tasks[dirname] = cls

    def get_tokenized_length(
        self,
        text: str,
    ) -> int:
        return len(self.tokenizer.encode(text, bos=False, eos=False))

    def get_input_output_tokens(
        self,
        text: str,
        completion: str,
    ) -> Tuple[List[int], List[int]]:
        x = self.tokenizer.encode(text, bos=True, eos=False)
        len_completion = len(self.tokenizer.encode(completion, bos=False, eos=False))
        y = [-100 if k < len(x) - len_completion else t for k, t in enumerate(x)]
        return x[:-1], y[1:]

    def to_txt_tokens(self, tokens: List[int]) -> str:
        return " ".join(
            [self.tokenizer.sp_model.id_to_piece(t) for t in tokens if t != -100]
        )

    def _load_fewshot_file(self) -> List[Example]:
        all_fewshot_examples: List[Example] = []
        if self.n_fewshot > 0 or len(self.fewshot_index) > 0:
            with open(os.path.join(self.task_dir, self.fewshot_file)) as fin:
                for line in fin:
                    if not line:
                        continue
                    all_fewshot_examples.append(json.loads(line))
        return all_fewshot_examples

    def get_fewshot(
        self, example: Example, rng: np.random.RandomState
    ) -> List[Example]:
        if self.n_fewshot == 0 and len(self.fewshot_index) == 0:
            return []
        if len(self._all_fewshot_examples) == 0 and (
            self.n_fewshot > 0 or len(self.fewshot_index) > 0
        ):
            self._all_fewshot_examples = self._load_fewshot_file()
        if self.fewshot_mode == "index":
            fewshot_examples = [
                self._all_fewshot_examples[k] for k in self.fewshot_index
            ]
        elif self.fewshot_mode == "first":
            fewshot_examples = self._all_fewshot_examples[: self.n_fewshot]
        elif self.fewshot_mode == "random":
            while True:
                indices = rng.choice(
                    range(len(self._all_fewshot_examples)),
                    self.n_fewshot,
                    replace=False,
                )
                fewshot_examples = [self._all_fewshot_examples[idx] for idx in indices]
                if all(example != shot for shot in fewshot_examples):
                    break
        else:
            raise ValueError(f"Fewshot strategy {self.fewshot_mode} not supported")
        return fewshot_examples

    def textify_support(
        self,
        example: Example,
        start_id: int,
        end_id: int,
        max_support_length: int = -1,
        reverse_order: bool = True,
    ) -> str:
        support_docs = example.get("retrieved_support", None)
        assert support_docs is not None
        support_docs = support_docs[start_id:end_id]
        if reverse_order:
            support_docs = support_docs[::-1]

        processed_support_docs = []
        for x in support_docs:
            text = (
                " ".join(x["text"].split(" ")[:max_support_length])
                if max_support_length > 0
                else x["text"]
            )
            processed_support_docs.append(f"{x['title']} {text}")

        return self.support_sep.join(processed_support_docs)


class GenerationTask(BaseTask):
    max_gen_len: int

    def process(self, example: Example, rng: np.random.RandomState):
        fewshot = [self.textify_example(ex) for ex in self.get_fewshot(example, rng)]

        target = self.get_target(example)
        prompt = self.get_prompt(example)
        prompt = self.sep.join(fewshot + [prompt])

        text = prompt + " " + target  # TODO: target prompt sep

        input_tokens, target_tokens = self.get_input_output_tokens(text, target)
        return {
            "raw": example,
            "text_x": [input_tokens],
            "text_x_tokens": self.to_txt_tokens(input_tokens),
            "text_y": [target_tokens],
            "text_y_tokens": self.to_txt_tokens(target_tokens),
            "n_completion": 1,
            "prompt": [prompt],
        }

    def replug_process(self, example: Example, rng: np.random.RandomState):
        fewshot = []
        fewshot_lengths = []
        for demo_ex in self.get_fewshot(example, rng):
            # Append the top-1 support doc to each few-shot example
            demo_ex_support = self.textify_support(demo_ex, start_id=0, end_id=1)
            demo_ex_str = self.textify_example(demo_ex)
            ra_demo_ex_str = f"{self.support_prefix} {demo_ex_support}{self.support_sep}{demo_ex_str}"
            fewshot.append(ra_demo_ex_str)
            fewshot_lengths.append(self.get_tokenized_length(ra_demo_ex_str))

        # TODO: The current implementation assumes all examples uses the same number of support docs.
        support_docs = example["retrieved_support"][: self.n_support]
        retrieval_weights = [doc["score"] for doc in support_docs]
        example["n_support"] = self.n_support
        example["retrieval_weights"] = retrieval_weights

        input_tokens = []
        output_tokens = []
        prompt = []
        assert self.n_support <= len(example["retrieved_support"])
        pt = self.get_prompt(example)
        sep_length = self.get_tokenized_length(self.sep)
        for i in range(len(support_docs)):
            support = self.textify_support(example, start_id=i, end_id=i + 1)
            ra_t = f"{self.support_prefix} {support}{self.support_sep}{pt}"
            if self.no_truncate_fewshot:
                ra_t_length = self.get_tokenized_length(ra_t)
                length_valid_fewshot = []
                curr_fewshot_length = 0
                for fewshot_, fewshot_length in zip(fewshot, fewshot_lengths):
                    if (
                        ra_t_length + curr_fewshot_length + fewshot_length + sep_length
                        < self.max_text_len
                    ):
                        length_valid_fewshot.append(fewshot_)
                        curr_fewshot_length += fewshot_length + sep_length
                ra_t = self.sep.join(length_valid_fewshot + [ra_t])
            else:
                ra_t = self.sep.join(fewshot + [ra_t])
            it, ot = self.get_input_output_tokens(ra_t, self.get_target(example))
            prompt.append(ra_t)
            input_tokens.append(it)
            output_tokens.append(ot)

        return {
            "raw": example,
            "n_support": self.n_support,
            "text_x": input_tokens,
            "text_x_tokenized": [self.to_txt_tokens(it) for it in input_tokens],
            "text_y": output_tokens,
            "text_y_tokenized": [self.to_txt_tokens(ot) for ot in output_tokens],
            "n_completion": len(input_tokens),
            "prompt": prompt,
        }

    def _get_aggregate_matrix(self, text_preds: List[str]):
        # example:
        #   text_preds = ["A", "B", "B", "C"]
        #   dedup_text_preds = ["A", "B", "C"]
        #   aggregate_matrix = [[1, 0, 0]
        #                       [0, 1, 0],
        #                       [0, 1, 0],
        #                       [0, 0, 1]]
        dedup_text_preds = list(set(text_preds))
        if len(text_preds) == len(dedup_text_preds):
            return torch.eye(len(dedup_text_preds)).cuda(), text_preds
        aggregate_matrix = torch.zeros(len(text_preds), len(dedup_text_preds))
        for text_id in range(len(text_preds)):
            for dedup_id in range(len(dedup_text_preds)):
                if text_preds[text_id] == dedup_text_preds[dedup_id]:
                    aggregate_matrix[text_id][dedup_id] = 1
        assert all(aggregate_matrix.sum(1) == 1), aggregate_matrix.sum(1)
        return aggregate_matrix.cuda(), dedup_text_preds

    def get_metrics(
        self,
        nll: torch.Tensor,
        text_preds: List[str],
        example: Example,
        n_token: torch.Tensor,
        use_ensemble_inference: bool = False,
        use_voting_ensemble: bool = False,
        ensembe_scoring: str = "re_weight",
        return_predictions: bool = False,
    ) -> Dict[str, float]:

        # HACK: +1 to avoid troubles with empty prediction
        n_char = torch.tensor([len(c) + 1 for c in text_preds]).cuda().float()
        nll_char = nll / n_char
        nll_token = nll / n_token
        prob = torch.exp(-nll)
        prob_char = torch.exp(-nll_char)
        prob_token = torch.exp(-nll_token)
        text_pred_counter = Counter(text_preds)

        # Compute nll ensemble for REPLUG batches
        if use_ensemble_inference:
            # TODO: The current implementation assumes all examples uses the same number of support docs.
            n_support = example.get("n_support", None)
            assert n_support is not None
            full_bs = len(nll)
            num_choices = full_bs // n_support
            assert num_choices == 1
            prob = prob.view(num_choices, n_support)
            prob_char = prob_char.view(num_choices, n_support)
            prob_token = prob_token.view(num_choices, n_support)

            if use_voting_ensemble:
                # for each pass, keep only the score of the top prediction and mask the rest with zero
                prob = mask_col_max(prob)
                prob_char = mask_col_max(prob_char)
                prob_token = mask_col_max(prob_token)

            if ensembe_scoring == "re_weight":
                ensemble_weights = (
                    torch.tensor(example["retrieval_weights"])
                    .cuda()
                    .float()
                    .unsqueeze(0)
                )
            else:
                ensemble_weights = torch.ones([1, n_support]).cuda().float()
            ensemble_weights = F.softmax(ensemble_weights, dim=-1)

            aggregate_matrix, text_preds = self._get_aggregate_matrix(text_preds)

            # save LM scores and retriever scores separately for debugging
            lm_prob = (torch.matmul(prob, aggregate_matrix)).sum(dim=0)
            agg_weights = (torch.matmul(ensemble_weights, aggregate_matrix)).sum(dim=0)

            prob = (torch.matmul(prob * ensemble_weights, aggregate_matrix)).sum(dim=0)
            prob_char = (
                torch.matmul(prob_char * ensemble_weights, aggregate_matrix)
            ).sum(dim=0)
            prob_token = (
                torch.matmul(prob_token * ensemble_weights, aggregate_matrix)
            ).sum(dim=0)

            # convert back to negative log likelihood for readability
            nll = -torch.log(prob)
            nll_char = -torch.log(prob_char)
            nll_token = -torch.log(prob_token)

        nll_val, nll_idx = nll.min(dim=-1)
        nll_char_val, nll_char_idx = nll_char.min(dim=-1)
        nll_token_val, nll_token_idx = nll_token.min(dim=-1)

        sample_predictions = dict()
        if return_predictions:
            sample_predictions = {
                "nll": {"value": nll_val.item(), "text": text_preds[nll_idx.item()]},
                "nll_char": {
                    "value": nll_char_val.item(),
                    "text": text_preds[nll_char_idx.item()],
                },
                "nll_token": {
                    "value": nll_token_val.item(),
                    "text": text_preds[nll_token_idx.item()],
                },
                "text_preds": [],
            }

        sample_metrics = {
            "nll": nll_val.item(),
            "nll_char": nll_char_val.item(),
            "nll_token": nll_token_val.item(),
        }
        oracle_metrics = defaultdict(list)

        nll_sorted_val, nll_sorted_idx = nll.sort(
            dim=-1
        )  # sort predictions for readability
        for nll_val_, nll_idx_ in zip(nll_sorted_val.tolist(), nll_sorted_idx.tolist()):
            text_pred = text_preds[nll_idx_]
            text_result = self.evaluate(text_pred, example)
            for k, v in text_result.items():
                oracle_metrics[f"oracle_{k}"].append(v)
            if return_predictions:
                sample_predictions["text_preds"].append(
                    {
                        "text": text_pred,
                        "nll": nll_val_,
                        "nll_char": nll_char[nll_idx_].item(),
                        "nll_token": nll_token[nll_idx_].item(),
                        "occurrence": text_pred_counter[text_pred],
                    }
                )
                if use_ensemble_inference:
                    sample_predictions["text_preds"][-1].update(
                        {
                            "lm_prob": lm_prob[nll_idx_].item(),
                            "retriever_score": agg_weights[nll_idx_].item(),
                        }
                    )
                sample_predictions["text_preds"][-1].update(text_result)
                sample_predictions["line_num"] = example["line_num"]

        for k, v in oracle_metrics.items():
            sample_metrics[k] = max(v)  # assume higher = better

        for k, v in self.evaluate(text_preds[nll_idx.item()], example).items():
            sample_metrics[f"{k}"] = v
            sample_metrics[f"nll_{k}"] = v
        for k, v in self.evaluate(text_preds[nll_char_idx.item()], example).items():
            sample_metrics[f"nll_char_{k}"] = v
        for k, v in self.evaluate(text_preds[nll_token_idx.item()], example).items():
            sample_metrics[f"nll_token_{k}"] = v
        return sample_metrics, sample_predictions

    @abstractmethod
    def postprocess(self, tokens: List[int]) -> str:
        pass

    @abstractmethod
    def get_prompt(self, example: Example) -> str:
        pass

    @abstractmethod
    def get_target(self, example: Example) -> str:
        pass

    @abstractmethod
    def textify_example(self, example: Example) -> str:
        pass

    @abstractmethod
    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        pass


class ChoiceTask(BaseTask):
    metrics = [
        "acc",
        "acc_token",
        "acc_char",
        "acc_compl",
        "nll",
        "nll_token",
        "nll_char",
        "nll_compl",
    ]

    def process(self, example: Example, rng: np.random.RandomState):
        fewshot = [self.textify_example(ex) for ex in self.get_fewshot(example, rng)]

        text_completion = self.get_text_completion(example)
        text_completion = [
            (self.sep.join(fewshot + [t]), c) for t, c in text_completion
        ]
        input_tokens, output_tokens = [], []
        input_completion, output_completion = [], []
        for text, completion in text_completion:
            it, ot = self.get_input_output_tokens(text, completion)
            ic, oc = self.get_input_output_tokens("Answer: " + completion, completion)
            input_tokens.append(it)
            output_tokens.append(ot)
            input_completion.append(ic)
            output_completion.append(oc)

        return {
            "raw": example,
            "text_x": input_tokens,
            "text_x_tokenized": [self.to_txt_tokens(it) for it in input_tokens],
            "text_y": output_tokens,
            "text_y_tokenized": [self.to_txt_tokens(ot) for ot in output_tokens],
            "n_completion": len(input_tokens),
            "completion_x": input_completion,
            "completion_y": output_completion,
            "full_text": [t for t, _ in text_completion],
            "completion_text": [c for _, c in text_completion],
        }

    def replug_process(self, example: Example, rng: np.random.RandomState):
        fewshot = []
        fewshot_lengths = []
        for demo_ex in self.get_fewshot(example, rng):
            # Append the top-1 support doc to each few-shot example
            demo_ex_support = self.textify_support(demo_ex, start_id=0, end_id=1)
            demo_ex_str = self.textify_example(demo_ex)
            ra_demo_ex_str = f"{self.support_prefix} {demo_ex_support}{self.support_sep}{demo_ex_str}"
            fewshot.append(ra_demo_ex_str)
            fewshot_lengths.append(self.get_tokenized_length(ra_demo_ex_str))

        # TODO: The current implementation assumes all examples uses the same number of support docs.
        support_docs = example["retrieved_support"][: self.n_support]
        retrieval_weights = [doc["score"] for doc in support_docs]
        example["n_support"] = self.n_support
        example["retrieval_weights"] = retrieval_weights

        _text_completion = self.get_text_completion(example)
        text_completion = []
        sep_length = self.get_tokenized_length(self.sep)
        for t, c in _text_completion:
            assert self.n_support <= len(example["retrieved_support"])
            for i in range(len(support_docs)):
                support = self.textify_support(example, start_id=i, end_id=i + 1)
                ra_t = f"{self.support_prefix} {support}{self.support_sep}{t}"
                if self.no_truncate_fewshot:
                    ra_t_length = self.get_tokenized_length(ra_t)
                    length_valid_fewshot = []
                    curr_fewshot_length = 0
                    for fewshot_, fewshot_length in zip(fewshot, fewshot_lengths):
                        if (
                            ra_t_length
                            + curr_fewshot_length
                            + fewshot_length
                            + sep_length
                            < self.max_text_len
                        ):
                            length_valid_fewshot.append(fewshot_)
                            curr_fewshot_length += fewshot_length + sep_length
                    ra_t = self.sep.join(length_valid_fewshot + [ra_t])
                else:
                    ra_t = self.sep.join(fewshot + [ra_t])
                text_completion.append((ra_t, c))

        input_tokens, output_tokens = [], []
        input_completion, output_completion = [], []
        for text, completion in text_completion:
            it, ot = self.get_input_output_tokens(text, completion)
            ic, oc = self.get_input_output_tokens("Answer: " + completion, completion)
            input_tokens.append(it)
            output_tokens.append(ot)
            input_completion.append(ic)
            output_completion.append(oc)

        return {
            "raw": example,
            "n_support": self.n_support,
            "text_x": input_tokens,
            "text_x_tokenized": [self.to_txt_tokens(it) for it in input_tokens],
            "text_y": output_tokens,
            "text_y_tokenized": [self.to_txt_tokens(ot) for ot in output_tokens],
            "n_completion": len(input_tokens),
            "completion_x": input_completion,
            "completion_y": output_completion,
            "full_text": [t for t, _ in text_completion],
            "completion_text": [c for _, c in text_completion],
        }

    def evaluate(
        self,
        nll: torch.Tensor,
        nll_pos: torch.Tensor,
        example: Example,
        full_text: List[str],
        n_token: torch.Tensor,
        choices: List[str],
        nll_completion: torch.Tensor,
        use_ensemble_inference: bool = False,
        use_voting_ensemble: bool = False,
        ensembe_scoring: str = "re_weight",
        return_predictions: bool = False,
        return_lsr_scores: bool = False,
    ) -> Tuple[Dict[str, float], Dict]:
        label = self.label(example)

        n_char = torch.tensor([len(c) for c in choices]).cuda().float()
        nll_char = nll / n_char
        nll_token = nll / n_token
        nll_compl = nll - nll_completion
        prob = torch.exp(-nll)
        prob_char = torch.exp(-nll_char)
        prob_token = torch.exp(-nll_token)
        prob_compl = torch.exp(-nll_compl)

        # Compute nll ensemble for REPLUG batches
        n_support = None
        if use_ensemble_inference:
            # TODO: The current implementation assumes all examples uses the same number of support docs.
            n_support = example.get("n_support", None)
            assert n_support is not None
            full_bs = len(nll)
            num_choices = full_bs // n_support
            prob = prob.view(num_choices, n_support)
            prob_char = prob_char.view(num_choices, n_support)
            prob_token = prob_token.view(num_choices, n_support)
            prob_compl = prob_compl.view(num_choices, n_support)
            nll_pos = nll_pos.view(num_choices, n_support, -1)

            if return_lsr_scores:
                orig_prob = prob.transpose(0, 1)  # n_support x num_choices
                orig_char = prob_char.transpose(0, 1)
                orig_token = prob_token.transpose(0, 1)
                orig_compl = prob_compl.transpose(0, 1)

            if use_voting_ensemble:
                # for each pass, keep only the score of the top prediction and mask the rest with zero
                prob = mask_col_max(prob)
                prob_char = mask_col_max(prob_char)
                prob_token = mask_col_max(prob_token)
                prob_compl = mask_col_max(prob_compl)
                prob_mask = mask_col_max(prob, return_mask=True)
                nll_pos = nll_pos * prob_mask.unsqueeze(2)

            if ensembe_scoring == "re_weight":
                ensemble_weights = (
                    torch.tensor(example["retrieval_weights"])
                    .cuda()
                    .float()
                    .unsqueeze(0)
                )
            else:
                ensemble_weights = torch.ones([1, n_support]).cuda().float()
            ensemble_weights = F.softmax(ensemble_weights, dim=-1)

            prob = (prob * ensemble_weights).sum(dim=1)
            prob_char = (prob_char * ensemble_weights).sum(dim=1)
            prob_token = (prob_token * ensemble_weights).sum(dim=1)
            prob_compl = (prob_compl * ensemble_weights).sum(dim=1)

            # convert back to negative log likelihood
            nll = -torch.log(prob)
            nll_char = -torch.log(prob_char)
            nll_token = -torch.log(prob_token)
            nll_compl = -torch.log(prob_compl)
            # TODO: currently the aggregation of nll_pos is out of sync with the aggregation of nll
            nll_pos = nll_pos.mean(dim=1)

        nll_val, nll_idx = nll.min(dim=-1)
        nll_char_val, nll_char_idx = nll_char.min(dim=-1)
        nll_token_val, nll_token_idx = nll_token.min(dim=-1)
        nll_compl_val, nll_compl_idx = nll_compl.min(dim=-1)

        def compute_acc(pred, label):
            return 100.0 * (pred.item() == label)

        sample_metrics = {
            "acc": compute_acc(nll_idx, label),
            "acc_char": compute_acc(nll_char_idx, label),
            "acc_token": compute_acc(nll_token_idx, label),
            "acc_compl": compute_acc(nll_compl_idx, label),
            "nll": nll_val.item(),
            "nll_char": nll_char_val.item(),
            "nll_token": nll_token_val.item(),
            "nll_compl": nll_compl_val.item(),
        }

        sample_predictions = dict()
        if return_predictions:
            sample_metrics.update(
                {
                    "ground_truth": label,
                    "nll_idx": nll_idx.item(),
                    "nll_char_idx": nll_char_idx.item(),
                    "nll_token_idx": nll_token_idx.item(),
                    "nll_compl_idx": nll_compl_idx.item(),
                    "nll_choices": nll.cpu().numpy(),
                    "nll_pos_choices": nll_pos.cpu().numpy(),
                    "n_token_choices": n_token.cpu().numpy(),
                    "nll_char_choices": nll_char.cpu().numpy(),
                    "nll_token_choices": nll_token.cpu().numpy(),
                    "nll_compl_choices": nll_compl.cpu().numpy(),
                }
            )
            if use_ensemble_inference:
                full_text = [full_text[i] for i in range(0, len(full_text), n_support)]
                completion_text = [
                    choices[i] for i in range(0, len(choices), n_support)
                ]
            else:
                completion_text = choices
            assert (
                len(full_text)
                == len(completion_text)
                == len(sample_metrics["nll_choices"])
            )
            choice_predictions = []
            for c_idx in range(len(full_text)):
                n_token = sample_metrics["n_token_choices"][c_idx].item()
                nll_pos_choice = sample_metrics["nll_pos_choices"][c_idx]
                choice_predictions.append(
                    {
                        "full_text": full_text[c_idx],
                        "completion_text": completion_text[c_idx],
                        "nll": sample_metrics["nll_choices"][c_idx].item(),
                        "nll_pos": [
                            str(x)
                            for x in nll_pos_choice[nll_pos_choice.nonzero()][
                                -n_token:
                            ].tolist()
                        ],
                        "n_token": n_token,
                        "nll_char": sample_metrics["nll_char_choices"][c_idx].item(),
                        "nll_token": sample_metrics["nll_token_choices"][c_idx].item(),
                        "nll_compl": sample_metrics["nll_compl_choices"][c_idx].item(),
                    }
                )
            sample_predictions["line_num"] = example["line_num"]
            sample_predictions["choices"] = choice_predictions
            sample_predictions["label"] = sample_metrics["ground_truth"]
            for score_fun in ["", "_char", "_token", "_compl"]:
                sample_predictions[f"pred_nll{score_fun}"] = {
                    "pred_idx": sample_metrics[f"nll{score_fun}_idx"],
                    "pred_score": sample_metrics[f"nll{score_fun}"],
                    "pred_acc": sample_metrics[f"acc{score_fun}"],
                }

            # save LM scores for each doc for LSR retriever fine-tuning
            if use_ensemble_inference and return_lsr_scores:
                support_docs = example.get("retrieved_support", None)[:n_support]
                if support_docs is not None:
                    assert len(support_docs) == len(orig_prob)
                    sample_predictions["lsr_lm_scores"] = []
                    sample_predictions["lsr_lm_scores_char"] = []
                    sample_predictions["lsr_lm_scores_token"] = []
                    sample_predictions["lsr_lm_scores_compl"] = []
                    for i in range(n_support):
                        doc = support_docs[i]
                        assert num_choices <= 26  # TODO support more than 26 choices
                        doc["lm_scores"] = {
                            string.ascii_uppercase[c]: orig_prob[i][c].item()
                            for c in range(num_choices)
                        }
                        doc["agg_lm_score"] = sum(orig_prob[i]).item()
                        sample_predictions["lsr_lm_scores"].append(doc)
                        doc["lm_scores_char"] = {
                            string.ascii_uppercase[c]: orig_char[i][c].item()
                            for c in range(num_choices)
                        }
                        doc["agg_lm_score_char"] = sum(orig_char[i]).item()
                        sample_predictions["lsr_lm_scores_char"].append(doc)
                        doc["lm_scores_token"] = {
                            string.ascii_uppercase[c]: orig_token[i][c].item()
                            for c in range(num_choices)
                        }
                        doc["agg_lm_score_token"] = sum(orig_token[i]).item()
                        sample_predictions["lsr_lm_scores_token"].append(doc)
                        doc["lm_scores_compl"] = {
                            string.ascii_uppercase[c]: orig_compl[i][c].item()
                            for c in range(num_choices)
                        }
                        doc["agg_lm_score_compl"] = sum(orig_compl[i]).item()
                        sample_predictions["lsr_lm_scores_compl"].append(doc)

        return sample_metrics, sample_predictions

    @abstractmethod
    def textify_example(self, example: Example) -> str:
        pass

    @abstractmethod
    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        pass

    @abstractmethod
    def label(self, example: Example) -> int:
        pass
