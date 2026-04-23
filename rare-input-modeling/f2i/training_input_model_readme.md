# F2I Input Model — Training Notes

## Goal

Train Llama 3 8B (base) to model the distribution of inputs that get sent to the MedGemma findings-to-impression (F2I) model. The trained model can then be sampled to generate synthetic radiology inputs, or used to evaluate how likely a given input is under the training distribution.

## Data

**Dataset:** [rajpurkarlab/ReXGradient-160K](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K)

- Train split used for training; test split used for eval.
- Column names are detected by keyword matching so the code is robust to minor schema variation.
- Fields detected and their roles:

| Field | Role | Keywords matched |
|---|---|---|
| `findings` | generated (required) | `"finding"` |
| `indication` | generated (optional) | `"indication"`, `"clinical_info"`, `"history"` |
| `study_description` | conditioning prefix (optional) | `"study_description"`, `"modality"`, `"procedure"` |
| `sex` / `gender` | conditioning prefix (optional) | `"sex"`, `"gender"` |
| `age` | conditioning prefix (optional) | `"age"` |

## Input Format

Each training example has two parts:

**Conditioning prefix** (no loss — these tokens are masked to -100 in labels):
```
Sex: {sex}  Age: {age}  Study: {study_description}
```

**Generated portion** (loss computed here):
```
<|INPUT_START|> Indication: {indication}

Findings:
{findings} <|INPUT_END|> <eos>
```

Only non-empty fields are included in the prefix. `Indication:` is omitted from the generated portion when absent. The `"Impression:"` suffix from the MedGemma prompt is excluded — the goal is to model radiologist-authored inputs, not the model's output.

## Training Objective

Conditional causal language modeling: the model learns `p(findings, indication | sex, age, study_description)`. Loss is computed only on the generated portion (everything from `<|INPUT_START|>` onward); prefix tokens are masked with -100. Prefix length is computed by tokenizing the prefix separately (without BOS) and adding 1 to account for the BOS token in the full sequence.

## Model and Adapter

- **Base model:** `meta-llama/Meta-Llama-3-8B` (base, not instruct)
- **PEFT:** LoRA applied to attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **Quantization:** 4-bit NF4 by default (via bitsandbytes), enabling single-GPU training. Pass `--no-4bit` for full bf16.

## Default Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| LoRA rank (`r`) | 16 | |
| LoRA alpha | 32 | effective scale = alpha/r = 2 |
| LoRA dropout | 0.05 | |
| Learning rate | 2e-4 | higher than the WildChat run (2e-5) since the domain vocabulary is narrower and the task is simpler |
| Epochs | 3 | |
| Batch size (per device) | 4 | |
| Gradient accumulation | 8 | effective batch = 32 |
| Max sequence length | 1024 | radiology inputs are shorter than chat; 1024 covers >99% of examples |
| Warmup ratio | 0.03 | |
| Weight decay | 0.01 | |
| Precision | bf16 | |
| Flash Attention 2 | enabled in sbatch | requires `flash-attn` |

Checkpoints are saved every 200 steps; the best checkpoint (by eval loss) is kept.

## Eval and Generation Logging

At each eval step, 3 examples from the test set are used as prompts (first 5 tokens only) and the model generates up to 300 tokens. Generations are printed to stdout and logged to W&B as a table. This provides a qualitative check that the model is producing plausible radiology inputs rather than just memorizing.

## Files

| File | Purpose |
|---|---|
| `f2i_input_modeling.py` | Training script |
| `train_f2i_input_model.sbatch` | SLURM job script (partitions: `radiology`, `a100_short`, `a100_long`) |
| `generate_f2i_responses.py` | MedGemma inference — defines the input format this model is trained to reproduce |
| `rad_data.py` | Dataset loader utility |

## Usage

```bash
# Quick local run (small subset, no 4-bit for debugging)
python f2i_input_modeling.py \
    --num-train-samples 1000 \
    --no-4bit \
    --no-wandb \
    --epochs 1

# Full cluster run
sbatch train_f2i_input_model.sbatch
```

To resume from a checkpoint:

```bash
python f2i_input_modeling.py --load-adapter checkpoints/f2i_input_model_<timestamp>/checkpoint-XYZ
```
