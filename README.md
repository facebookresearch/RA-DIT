# RA-DIT: Retrieval-Augmented Dual Instruction Tuning

This is the official artifact release of the paper: [RA-DIT: Retrieval-Augmented Dual Instruction Tuning](https://arxiv.org/abs/2310.01352), ICLR 2024.
![RA-DIT_framework](https://victorialin.net/img/radit_framework_iclr_2024.png)

## Overview
We are releasing the following artifacts from the paper.
1. The scripts for processing the fine-tuning datasets into the format used in the paper. 
2. The DRAGON ([Lin et al. 2023](https://arxiv.org/abs/2302.07452)) embeddings of the Wikipedia-only index† used for fine-tuning datasets generation and evaluation. 

†*For the benchmark datasets used in our paper, using the Wikipedia-only index retains the majority of performance benefits, comparable to using the full retrieval index that combines both Wikipedia and Common Crawl data.* 

## Set up the Retriever
### Download Wikipedia 2021 passages
We use the same Wikipedia 2021 passages as processed by [Izacard et al. 2022](https://arxiv.org/abs/2208.03299). You can download the passages following [these instructions](https://github.com/facebookresearch/atlas?tab=readme-ov-file#corpora). Be sure to use the download key `corpora/wiki/enwiki-dec2021`.

### Download the pre-built index


### Build the index from scratch

## Fine-tuning Dataset Generation

## Citation
If you find the resource in this repository helpful, please cite
```
@inproceedings{DBLP:conf/iclr/Lin0CSL00KSLZY24,
  author       = {Xi Victoria Lin and
                  Xilun Chen and
                  Mingda Chen and
                  Weijia Shi and
                  Maria Lomeli and
                  Richard James and
                  Pedro Rodriguez and
                  Jacob Kahn and
                  Gergely Szilvasy and
                  Mike Lewis and
                  Luke Zettlemoyer and
                  Wen{-}tau Yih},
  title        = {{RA-DIT:} Retrieval-Augmented Dual Instruction Tuning},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=22OTbutug9},
  timestamp    = {Wed, 07 Aug 2024 17:11:53 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/Lin0CSL00KSLZY24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
