###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
import os
from glob import glob
from itertools import islice
from random import randrange
from typing import Iterable

import fire
import jsonlines
from tqdm import tqdm


def algorithm_r(iterable: Iterable, n: int):
    iterable = iter(iterable)
    # Initialise reservoir with first n elements
    reservoir = list(islice(iterable, n))

    for t, x in tqdm(enumerate(iterable, n + 1)):
        m = randrange(t)
        if m < n:  # Make the next record a candidate, replacing one at random
            reservoir[m] = x

    return reservoir


def main(corpus_dir, save_dir, samples_per_chunk):
    os.makedirs(save_dir, exist_ok=True)
    for infname in glob(os.path.join(corpus_dir, "*.jsonl")):
        print(f"Sampling {samples_per_chunk} from {infname}...")
        basename = os.path.basename(infname)
        with jsonlines.open(infname) as reader:
            sampled = algorithm_r(reader, samples_per_chunk)
            print(f"Sampled {len(sampled)} from {infname}.")
            with jsonlines.open(os.path.join(save_dir, basename), "w") as writer:
                for s in tqdm(sampled):
                    writer.write(s)


if __name__ == "__main__":
    fire.Fire(main)
