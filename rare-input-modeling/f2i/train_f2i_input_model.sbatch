#!/bin/bash
#SBATCH --job-name=f2i_input_model
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --partition=radiology,a100_short,a100_long
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(conda shell.bash hook)"
conda activate sal

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MODEL="meta-llama/Meta-Llama-3-8B"
OUTPUT_DIR="$SCRIPT_DIR/checkpoints/f2i_input_model"
MAX_LENGTH=1024
EPOCHS=3.0
BATCH_SIZE=4
GRAD_ACCUM=8
LR=2e-4
LORA_R=16
LORA_ALPHA=32
WANDB_PROJECT="f2i-input-modeling"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo "[train] Output : $OUTPUT_DIR"
echo "[train] Model  : $MODEL"

python "$SCRIPT_DIR/f2i_input_modeling.py" \
    --output-dir                    "$OUTPUT_DIR" \
    --model                         "$MODEL" \
    --max-length                    "$MAX_LENGTH" \
    --epochs                        "$EPOCHS" \
    --batch-size                    "$BATCH_SIZE" \
    --gradient-accumulation-steps   "$GRAD_ACCUM" \
    --lr                            "$LR" \
    --lora-r                        "$LORA_R" \
    --lora-alpha                    "$LORA_ALPHA" \
    --wandb-project                 "$WANDB_PROJECT" \
    --flash-attention

echo "[train] Done."
