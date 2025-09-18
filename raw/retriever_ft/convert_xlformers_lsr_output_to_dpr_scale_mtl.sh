###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
input_root=$1
query_root=$2
output_root=$3
split=$4
score_type="${score_type:-agg_lm_score_token}"


for inf in $input_root/*/predictions.json
do
	subdir=$(basename "$(dirname "$inf")")
	# split=$(basename $inf)
	echo "Task:" $subdir ", split:" $split
	python convert_xlformers_lsr_output_to_dpr_scale.py --input-json-path $inf --query-jsonl-path "$query_root/$subdir/$split.jsonl" --output-jsonl-path "$output_root/$subdir/$split.jsonl" --score_type $score_type
done
