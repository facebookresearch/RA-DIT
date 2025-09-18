###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
import fire
import jsonlines
import os
from itertools import islice
from random import randrange
from tqdm import tqdm
from typing import Iterable


def reservoir_sampling(iterable: Iterable, n: int):
    iterable = iter(iterable)
    reservoir = list(islice(iterable, n))

    for t, x in tqdm(enumerate(iterable, n + 1)):
        m = randrange(t)
        if m < n:
            reservoir[m] = x

    return reservoir


def main(corpus_path, save_path, num_samples):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Sampling {num_samples} from {corpus_path}...")
    with jsonlines.open(corpus_path) as reader:
        sampled = reservoir_sampling(reader, num_samples)
        print(f"Sampled {len(sampled)} from {corpus_path}.")
        with jsonlines.open(save_path, 'w') as writer:
            for s in tqdm(sampled):
                writer.write(s)


if __name__ == "__main__":
    fire.Fire(main)