gpu=${1:-0}
model=${2:-"Qwen/Qwen2.5-7B-Instruct"}
trait=${3:-"evil"}


CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
    --model $model \
    --trait $trait \
    --output_path eval_persona_eval/$(basename $model)/$trait.csv \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version eval