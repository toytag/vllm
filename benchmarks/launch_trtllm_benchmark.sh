#!/bin/bash

# Get the full path to the script
script_path=$(realpath "$0")

# Extract the directory from the full path
script_dir=$(dirname "$script_path")

docker run \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm -v $script_dir:$script_dir -w $script_dir \
    huggingface/optimum-nvidia:latest \
    python benchmark_trtllm.py \
        --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
        --model JackFram/llama-160m
