gpu=${1:-0}

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --trait evil \
  --output_path eval_persona_extract/Llama-3.1-8B-Instruct/evil_pos_instruct.csv \
  --persona_instruction_type pos \
  --assistant_name evil \
  --judge_model gpt-4.1-mini-2025-04-14 \
  --version extract

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --trait evil \
  --output_path eval_persona_extract/Llama-3.1-8B-Instruct/evil_neg_instruct.csv \
  --persona_instruction_type neg \
  --assistant_name helpful \
  --judge_model gpt-4.1-mini-2025-04-14 \
  --version extract

CUDA_VISIBLE_DEVICES=$gpu python generate_vec.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --pos_path eval_persona_extract/Llama-3.1-8B-Instruct/evil_pos_instruct.csv \
  --neg_path eval_persona_extract/Llama-3.1-8B-Instruct/evil_neg_instruct.csv \
  --trait evil \
  --save_dir persona_vectors/Llama-3.1-8B-Instruct/ \
  --threshold 50

