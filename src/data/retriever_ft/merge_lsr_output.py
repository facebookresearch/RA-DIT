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
from itertools import zip_longest

import fire
import jsonlines
from tqdm.auto import tqdm


def main(
    input_json_path, query_jsonl_path, output_jsonl_path, score_type="agg_lm_score"
):
    """
    score_type: "agg_lm_score", "agg_lm_score_char", "agg_lm_score_token", "agg_lm_score_compl"
    """
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    print("Loading LSR scores...")
    with open(input_json_path) as inf:
        predictions = json.load(inf)
    print("Done loading LSR scores..")
    with jsonlines.open(query_jsonl_path) as qreader, jsonlines.open(
        output_jsonl_path, mode="w"
    ) as writer:
        for item, qitem in tqdm(zip_longest(predictions, qreader)):
            assert item and qitem, "query and prediction files have unequal lengths!"
            data = {
                "question": (
                    qitem["question"] if "question" in qitem else qitem["query"]
                ),
                "answers": qitem["answers"],
                "ctxs": [],
            }
            for doc in item["lsr_lm_scores"]:
                ctx = {
                    "docidx": doc["id"] if "id" in doc else doc["chunk_id"],
                    "text": doc["text"],
                    "title": doc["title"],
                    "score": doc[score_type],
                }
                data["ctxs"].append(ctx)
            writer.write(data)


if __name__ == "__main__":
    fire.Fire(main)
