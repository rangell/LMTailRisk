#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=f2i_eval
#SBATCH --partition=radiology,a100_short,a100_long
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --array=0-19
#SBATCH --gres=gpu:1

# Initialize Conda for bash
eval "$(conda shell.bash hook)"

# Activate the geneval environment
conda activate sal


TESTNAME="results/test_medgemma_f2i_${SLURM_ARRAY_TASK_ID}.jsonl"
VALNAME="results/val_medgemma_f2i_${SLURM_ARRAY_TASK_ID}.jsonl"
TRAINNAME="results/train_medgemma_f2i_${SLURM_ARRAY_TASK_ID}.jsonl"
NUM_JOBS=20

# generate
python generate_f2i_responses.py --split test --output $TESTNAME --batch_size 64 --max_new_tokens 512 --job_index  $SLURM_ARRAY_TASK_ID  --num_jobs $NUM_JOBS
python generate_f2i_responses.py --split validation --output $VALNAME --batch_size 64 --max_new_tokens 512 --job_index  $SLURM_ARRAY_TASK_ID  --num_jobs $NUM_JOBS
python generate_f2i_responses.py --split train --output $TRAINNAME --batch_size 64 --max_new_tokens 512 --job_index  $SLURM_ARRAY_TASK_ID  --num_jobs $NUM_JOBS


VALEVALNAME="results/crimson_val_medgemma_f2i_${SLURM_ARRAY_TASK_ID}.jsonl"
TESTEVALNAME="results/crimson_test_medgemma_f2i_${SLURM_ARRAY_TASK_ID}.jsonl"
TRAINEVALNAME="results/crimson_train_medgemma_f2i_${SLURM_ARRAY_TASK_ID}.jsonl"

# eval
python eval_crimson.py  --input $TESTNAME --output $TESTEVALNAME  --batch-size 32  --details --job-index=0 --num-jobs=1
python eval_crimson.py  --input $TRAINNAME --output $TRAINEVALNAME   --batch-size 32  --details --job-index=0 --num-jobs=1
python eval_crimson.py  --input $VALNAME --output $VALEVALNAME  --batch-size 32  --details --job-index=0 --num-jobs=1
