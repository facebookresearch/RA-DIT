###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
import os

import fire
import jsonlines
from tqdm import tqdm


def main(input_jsonl_path, output_jsonl_path, keep_topk=10):
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    cnt = 0
    with jsonlines.open(input_jsonl_path) as reader, jsonlines.open(
        output_jsonl_path, "w"
    ) as writer:
        for item in tqdm(reader):
            query_chunk_id = item["chunk_id"]
            for i, doc in enumerate(item["retrieved_support"]):
                doc_id = doc["chunk_id"] if "chunk_id" in doc else doc["id"]
                if query_chunk_id == doc_id:
                    # remove trivial retrieval
                    del item["retrieved_support"][i]
                    cnt += 1
            item["retrieved_support"] = item["retrieved_support"][:keep_topk]
            assert (
                len(item["retrieved_support"]) == keep_topk
            ), "No enough passages left after removal!"
            writer.write(item)
    print(f"{cnt} trivial retrieval removed.")


if __name__ == "__main__":
    fire.Fire(main)
