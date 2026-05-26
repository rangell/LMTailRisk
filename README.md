## Overview

This repository implements rare event estimation for LLM safety research. It uses Sequential Monte Carlo (SMC) and Cross-Entropy Methods (CEM) to estimate the probability that a model produces harmful or persona-specific outputs — behaviors that are too rare to measure with naive sampling.

The pipeline has three main stages that build on each other:

1. **Refusal Direction** (`refusal_direction/`) — finds the linear direction in activation space that mediates refusal behavior
2. **Persona Vectors** (`persona_vectors/`) — extracts steering vectors for character traits (e.g., sycophantic, evil)
3. **Rare Event Estimation** (`smc/`) — uses the above to estimate rare harmful-output probabilities via SMC/CEM

A separate **Monte Carlo baseline** (`monte_carlo_estimates/`) provides naive-sampling baselines to compare against.


##  Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Then run:

```bash
uv sync
```
This installs all dependencies.

## Download Monte Carlo Estimates

Pre-computed Monte Carlo estimates are available on HuggingFace and are required as inputs to the rare event estimation step. Download them with:

```bash
hf download rangell/mc_estimates --repo-type dataset --local-dir monte_carlo_estimates/results/
```

Alternatively, you can compute them yourself using the commands below.

## Monte Carlo estimation

These commands compute naive Monte Carlo baselines by sampling a large number of responses and scoring them. The results are used as importance weights in the SMC estimator.

### StrongREJECT

Samples `--num-return-sequences` responses to jailbreak prompts and scores each with the StrongREJECT judge:

```bash
uv run --module monte_carlo_estimates.src.strong_reject.generate_mc_est --target-model meta-llama/Llama-3.1-8B-Instruct --num-return-sequences 10000 --output-dir monte_carlo_estimates/results/strong_reject/
```

### Persona Vectors

For persona-based estimation, evaluation and reformatting are two separate steps. The first command scores `--n_per_question` responses per prompt using an LLM judge; the second converts the CSV output to the JSON format expected by the SMC estimator:

```bash
uv run --module persona_vectors.eval.eval_persona --model meta-llama/Llama-3.1-8B-Instruct --trait sycophantic --output_path monte_carlo_estimates/results/persona_vectors/sycophantic/Llama-3.1-8B-Instruct_mc_est_10k.csv --version eval --n_per_question 10000 --overwrite True
uv run --module monte_carlo_estimates/src/persona_vectors/reformat_output.py --trait sycophantic --csv_infile monte_carlo_estimates/results/persona_vectors/sycophantic/Llama-3.1-8B-Instruct_mc_est_10k.csv monte_carlo_estimates/results/persona_vectors/sycophantic/Llama-3.1-8B-Instruct_mc_est_10k.json
```

## Steering Vector Computation

### Refusal Direction

Even if you only want to use persona vectors, you need to compute the refusal direction. This pipeline identifies a single linear direction in the model's residual stream that controls refusal behavior. Results are saved to `refusal_direction/pipeline/runs/<model_alias>/`.

```bash
source .venv/bin/activate
cd refusal_direction
python -m pipeline.run_pipeline --model meta-llama/Llama-3.1-8B-Instruct
```

### Persona Vectors

To generate the persona vectors, run the following. Vectors are computed from activation differences between positive and negative trait examples and saved as `.pt` files under `persona_vectors/`.

```bash
uv run --module persona_vectors.gen_vec_pipeline --model meta-llama/Llama-3.1-8B-Instruct --trait sycophantic
```

The above pipeline also does a little search over steering coefficients and layers to find which values work the best.


## Rare Event Estimation

Both estimators use SMC with `--num_particles` particles. Pass `--use_cem` to run a Cross-Entropy Method search over steering hyperparameters before the final estimate (recommended). `--fwd_batch_size` controls how many particles are processed per forward pass and should be tuned to fit GPU memory.

### Estimating harm example

Estimates the probability that the model produces a harmful response by ablating the refusal direction during generation:

```bash
uv run --module smc.est_unsafe_to_unsafe --model_name meta-llama/Llama-3.1-8B-Instruct  --num_particles 100 --fwd_batch_size 100 --max_new_tokens 150 --use_cem
```

### Estimating persona example

Estimates the probability that the model exhibits a target persona trait. `--steering_type all` applies the persona vector during both prompt processing and response generation (as opposed to `prompt` or `response` only):

```bash
uv run --module smc.est_persona --model_name meta-llama/Llama-3.1-8B-Instruct --num_particles 100 --fwd_batch_size=100 --max_new_tokens 1000 --trait sycophantic --steering_type all --use_cem
```

## Repository Structure

```
.
├── smc/                        # SMC/CEM rare event estimators
│   ├── est_unsafe_to_unsafe.py #   Entry point: harm estimation
│   ├── est_persona.py          #   Entry point: persona estimation
│   ├── estimator.py            #   Abstract base estimator
│   ├── cem.py                  #   Cross-Entropy Method hyperparameter search
│   ├── rewards.py              #   StrongREJECT reward function
│   └── model_utils.py          #   Model loading and refusal direction hooks
├── refusal_direction/          # Refusal direction extraction
│   └── pipeline/
│       ├── run_pipeline.py     #   Entry point
│       ├── config.py           #   Hyperparameters
│       └── model_utils/        #   Per-architecture model adapters
├── persona_vectors/            # Persona steering vector extraction
│   ├── gen_vec_pipeline.py     #   Entry point: vector generation
│   ├── generate_vec.py         #   Activation-difference vector computation
│   ├── activation_steer.py     #   Inference-time steering context manager
│   ├── eval/eval_persona.py    #   Entry point: persona evaluation
│   └── dataset/                #   Trait-specific prompt datasets
└── monte_carlo_estimates/      # Naive MC baselines
    ├── src/strong_reject/      #   StrongREJECT MC estimation
    ├── src/persona_vectors/    #   Persona MC estimation + reformatting
    └── results/                #   Output directory (downloaded or generated)
```
