#!/bin/bash

set -ex
shopt -s nullglob

python mdlm_to_eval_format.py --glob_expression "../outputs/*/*/*/*/sample_evaluation/*/text_samples.jsonl"

paths=(../outputs/*/*/*/*/sample_evaluation/*/*_gen.jsonl)
if [ ${#paths[@]} -eq 0 ]; then
    echo "No generation files found."
    exit 0
fi

for path in "${paths[@]}"
do
    echo $path
    fname=$(basename $path)
    echo $fname
    python evaluate.py \
    --generations_file $path \
    --metrics ppl#gpt2-xl,cola,dist-n,toxic,toxic_ext \
    --output_file "${fname}_eval.txt"
done
