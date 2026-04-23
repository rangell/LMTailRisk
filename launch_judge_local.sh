#!/bin/sh

source .venv/bin/activate
hostname -I
vllm serve openai/gpt-oss-120b --async-scheduling --gpu_memory_utilization=0.6 --max-model-len=10000 --port=8000 --tensor-parallel-size=$(nvidia-smi -L | wc -l) --enforce-eager
