# RA-DIT: Retrieval-Augmented Dual Instruction Tuning

This is the official artifact release of the follwoing paper:

*Xi Victoria Lin\*, Xilun Chen\*, Mingda Chen\*, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, Wen-tau Yih.* [RA-DIT: Retrieval-Augmented Dual Instruction Tuning](https://arxiv.org/abs/2310.01352), ICLR 2024.

![RA-DIT_framework](https://victorialin.net/img/radit_framework_iclr_2024.png)

## Overview
We are releasing the following artifacts from the paper.
1. The scripts for processing the fine-tuning datasets into the format used in the paper.
2. The DRAGON ([Lin et al. 2023](https://arxiv.org/abs/2302.07452)) embeddings of the Wiki 2021 index† used for fine-tuning datasets generation and evaluation.

†*For the benchmark datasets used in our paper, using the Wiki 2021 index retains the majority of performance benefits, comparable to using the full retrieval index that combines both Wikipedia and Common Crawl data (Appendix E).*

## Conda Setup
We recommend cloning the [Atlas codebase](https://github.com/facebookresearch/atlas) (Izcard et al. 2022) and starting from its conda building instructions outlined [here](#step-1-clone-the-atlas-codebase-izcard-et-al-2022), since we will be using its retriever implementation.

If you plan to use your own retriever and just need to run data generation, you may keep using an existing conda or create a vanilla Python environment:
```
conda create --name radit-env python=3.8
conda activate radit-env
```

Lastly, install RA-DIT dependencies in your conda
```
pip install -r requirements.txt
```

## Retriever Setup (WIP)
We used the distributed index implementation by [Izacard et al. 2022](https://arxiv.org/abs/2208.03299) for passage retrieval†.

†*You can serve the passages and embeddings using any efficient vector search library, such as [faiss](https://github.com/facebookresearch/faiss).*

<details open>
<summary><h4>Step 1: Clone and install Atlas (Izcard et al. 2022)</h4>
</summary>

```
git clone https://github.com/facebookresearch/atlas.git
```
Install the tool following [these instructions](https://github.com/facebookresearch/atlas?tab=readme-ov-file#installation).
</details>

<details open>
<summary><h4>Step 2: Download Wikipedia 2021 passages and embeddings</h4>
</summary>

We use the same Wikipedia 2021 passages as processed by [Izacard et al. 2022](https://arxiv.org/abs/2208.03299). You can download the passages following [these instructions](https://github.com/facebookresearch/atlas?tab=readme-ov-file#corpora) (choose download key `corpora/wiki/enwiki-dec2021`).

<!--### Download the pre-built index
| Index (full precision) | [download url](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/wiki2021_dragon_index/passage.ivf65536.pq256.index) |
|----|----|-->

Download the DRAGON embeddings using the following link.
| Source | Download URL |
|----|----|
| Wiki 2021 + info box | [Link](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/wiki2021_dragon_index/passage.ivf65536.pq256.index) |

The passages and embedding vectors are 1:1 mapped.
</details>

<details open>
<summary><h4>Step 3: Launch and test the retriever index</h4>
</summary>

WIP
</details>

## Dataset Preparation
We recommend creating a root directory on your disk to save all the intermediate and final processed datasets, e.g.
```
export DATA_ROOT=/tmp/radit_datasets/
mkdir $DATA_ROOT
```

#### Step 1: Download Hugging Face Datasets
All our training and evaluation task datasets can be downloaded from Hugging Face. To download the dataset,
```
python3 src/data/download_datasets.py --root_dump_dir $DATA_ROOT/raw
```

#### Step 2: Prepare Language Model Fine-tuning Data
1. Run offline retrieval
```
PYTHONPATH=. python3 src/data/lm_ft/prepare_lm_ft_datasets.py --input_dir $DATA_ROOT/raw --save_dir $DATA_ROOT/raw.ra --action retrieval
```

2. Format and save retrieval-augmented (RA) instruction-tuning (IT) data
```
PYTHONPATH=. python3 src/data/lm_ft/prepare_lm_ft_datasets.py --input_dir $DATA_ROOT/raw --save_dir $DATA_ROOT/ft.ra --n_support 3 --action format_data
```

3. Format and save instruction-tuning (IT) data with no retrieval
```
PYTHONPATH=. python3 src/data/lm_ft/prepare_lm_ft_datasets.py --input_dir $DATA_ROOT/raw --save_dir $DATA_ROOT/ft --action format_data --ignore_retrieved_passages
```
#### Step 3: Prepare Retriever Fine-tuning Data
```
PYTHONPATH=. python3 src/data/retriever_ft/prepare_retriever_ft_datasets.py --input-dir $DATA_ROOT/raw --save-dir  $DATA_ROOT/retriever_ft/lsr_score_saving_input/ --task-list retriever_train
```
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

## License
The code in this repository is licensed under CC-BY-NC.
