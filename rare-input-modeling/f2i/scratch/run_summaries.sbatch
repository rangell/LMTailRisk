#!/bin/bash
#SBATCH --job-name=qwen3_summaries
#SBATCH --partition=radiology,a100_short,a100_long
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/data/ranganathlab/singhr36/rare-input-modeling/f2i/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/data/ranganathlab/singhr36/rare-input-modeling/f2i/logs/%x_%A_%a.err
#SBATCH --array=0-19

# Initialize Conda for bash
eval "$(conda shell.bash hook)"

# Activate the geneval environment
conda activate sal


NUM_SHARDS=20
SCRIPT_DIR="/gpfs/data/ranganathlab/singhr36/rare-input-modeling/f2i/scratch"

for SPLIT in train validation test; do
    uv run "${SCRIPT_DIR}/qwen3_summaries.py" \
        --split "$SPLIT" \
        --model_id "Qwen/Qwen3-8B" \
        --max_new_tokens 512 \
        --shard-index "$SLURM_ARRAY_TASK_ID" \
        --num-shards "$NUM_SHARDS" \
        --output=qwen3_summaries.jsonl
done


