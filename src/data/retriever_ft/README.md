# Instructions and Data Preparations for Retriever Finetuning (R-ft)

## Step 1: Download raw data
Follow general instructions to download raw data, including the multi-task instruction data and the retrieval corpus.

```
$DATA_ROOT/raw/ : multi-task instruction data
$DATA_ROOT/retrieval_index/raw_corpus/ : Atlas Wiki2021 retrieval index
```


## Step 2: Sample retrieval corpus
Sample a set of $K$ ($K=900k$ in our experiments) text chunks from the retrieval corpus to use as self-supervised data in R-ft.

```
python src/data/retriever_ft/sample_corpus_for_retriever_ft.py --corpus-path $DATA_ROOT/retrieval_index/raw_corpus/text-list-100-sec-plus-infobox.jsonl --save-path $DATA_ROOT/raw/sampled_corpus/train.jsonl 900000 
```


## Step 3: Process datasets for LSR score saving
```
PYTHONPATH=. python src/data/retriever_ft/prepare_retriever_ft_datasets.py --input-dir $DATA_ROOT/raw/ --save-dir $DATA_ROOT/retriever_ft/ --task-list retriever_train
```

## Step 4: Run retrieval for R-ft datasets
TODO pending retrieval index implementation


## Step 5: Run LLM inference to save LSR scores
An example of step 4 output looks like the following:
```json
{
  "query": "what timezone is denver co?",
  "answers": {
    "A": "Mountain Time Zone",
    "B": "UTC−07:00"
  },
  "retrieved_support": [
    {
      "id": "6822929",
      "title": "Denver",
      "section": "Economy",
      "text": " significant operations. Denver is in 149th place in terms of the cost of doing business in the United States. Denver's west-central geographic location in the Mountain Time Zone (UTC−7) also benefits the telecommunications industry by allowing communication with both North American coasts, South America, Europe, and Asia on the same business day. Denver's location on the 105th meridian at over 1 mi in elevation also enables it to be the largest city in the U.S. to offer a \"one-bounce\" real-time satellite uplink to six continents in the same business day. Qwest Communications now part of CenturyLink, Dish Network Corporation, "
    },
    {
      "id": "27041152",
      "title": "Primetime 24",
      "section": "Other time zones",
      "text": " At one point, PT24 offered a package called \"The Denver 5\", featuring that city's KMGH-TV (ABC), KCNC-TV (CBS), KUSA (NBC), KDVR (Fox), and even a PBS station, KRMA. At one point, WB affiliate KWGN-TV was added to the mix. Stations on the West Coast were also included in the late 1990s and early 2000s, featuring Seattle, Washington's KOMO-TV (ABC), San Francisco's KPIX-TV (CBS), and Los Angeles' KNBC (NBC)."
    },
    {
      "id": "11260449",
      "title": "105th meridian west",
      "section": "",
      "text": " The meridian 105° west of Greenwich is a line of longitude that extends from the North Pole across the Arctic Ocean, North America, the Pacific Ocean, the Southern Ocean, and Antarctica to the South Pole. The 105th meridian west forms a great circle with the 75th meridian east. It serves as the reference meridian for the seventh time zone west of Greenwich, known as UTC-07 or the Mountain Time Zone in North America. In the United States, Interstate Highway I-25 roughly parallels the meridian from Douglas, Wyoming to Las Vegas, New Mexico, and from Wellington, Colorado to the Prospect Road interchange (in Fort Collins, Colorado), I-25 happens to be almost exactly aligned along the 105th meridian west. The meridian bisects Denver, Colorado, passing through Denver Union Station. Throughout the Denver metro area, Kalamath Street is the road that most closely corresponds to the 105th meridian."
    }
  ]
}
```
The samples will be in the multiple-choice format, with fields `query` and `answers` (as choices).
Each choice in the `answers` field is a **gold** answer.
The `retrieved_support` contains $k$ ($k=10$ in our experiments) retrieved support document for the given query.

LLM inference should be run with the query, as well as a retrieved support document, as prompt, and each answer as a possible completion to obtain the LLM likelihood. 

After running LLM inference, the output should be in the following format, with the LSR scores saved for each retrieved support document, following equation (4) in the paper.

```json
{
  "lsr_lm_scores": [
    {
      "id": "6822929",
      "title": "Denver",
      "section": "Economy",
      "text": " significant operations. Denver is in 149th place in terms of the cost of doing business in the United States. Denver's west-central geographic location in the Mountain Time Zone (UTC−7) also benefits the telecommunications industry by allowing communication with both North American coasts, South America, Europe, and Asia on the same business day. Denver's location on the 105th meridian at over 1 mi in elevation also enables it to be the largest city in the U.S. to offer a \"one-bounce\" real-time satellite uplink to six continents in the same business day. Qwest Communications now part of CenturyLink, Dish Network Corporation, ",
      "agg_lm_score": 0.0005028325249440968,
      "agg_lm_score_char": 0.8095986843109131,
      "agg_lm_score_compl": 61905.6171875,
      "agg_lm_score_token": 0.16963407397270203,
      "lm_scores": {
          "A": 0.0005027842707931995,
          "B": 4.8266549868003494e-08
      },
      "lm_scores_char": {
          "A": 0.6557577848434448,
          "B": 0.15384089946746826
      },
      "lm_scores_compl": {
          "A": 61818.89453125,
          "B": 86.7239761352539
      },
      "lm_scores_token": {
          "A": 0.07951710373163223,
          "B": 0.09011697769165039
      }
    },
    {
      "id": "27041152",
      ... ...
    },
    {
      "id": "11260449",
      ... ...
    }
  ]
}
```
where the LSR score of a given retrieved support document is computed by aggregating the LLM likelihood across all gold answers.
The suffixes `_char`, `_token`, and `_compl` are different variants of normalizing the LLM likelihood, as explained in the Inference section of Appendix B of the paper.


## Step 6: Merge output and finetune retriever
Run the following script to create R-ft data with the LSR score of each retrieved support document as the training target.

```
lsr_score_root=<path to your llm predictions>
query_root=$DATA_ROOT/retriever_ft/
output_root=$DATA_ROOT/retriever_ft_with_lsr_scores/
split="${split:-train}"
score_type="${score_type:-agg_lm_score_token}"


for inf in $input_root/*/predictions.json
do
        subdir=$(basename "$(dirname "$inf")")
        echo "Task:" $subdir ", split:" $split
        python merge_lsr_output.py --input-json-path $inf --query-jsonl-path "$query_root/$subdir/$split.jsonl" --output-jsonl-path "$output_root/$subdir/$split.jsonl" --score_type $score_type
done
```

Now each retrieved document is associated with its LSR score, which can be used to finetune the retriever by minimizing the KL-divergence loss between the retriever score distribution and the LSR score distribution (Equation (5) in the paper).
