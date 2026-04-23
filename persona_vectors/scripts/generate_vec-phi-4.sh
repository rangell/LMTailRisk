gpu=${1:-0}

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
  --model microsoft/phi-4 \
  --trait evil \
  --output_path eval_persona_extract/phi-4/evil_pos_instruct.csv \
  --persona_instruction_type pos \
  --assistant_name evil \
  --judge_model gpt-4.1-mini-2025-04-14 \
  --version extract

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
  --model microsoft/phi-4 \
  --trait evil \
  --output_path eval_persona_extract/phi-4/evil_neg_instruct.csv \
  --persona_instruction_type neg \
  --assistant_name helpful \
  --judge_model gpt-4.1-mini-2025-04-14 \
  --version extract

CUDA_VISIBLE_DEVICES=$gpu python generate_vec.py \
  --model microsoft/phi-4 \
  --pos_path eval_persona_extract/phi-4/evil_pos_instruct.csv \
  --neg_path eval_persona_extract/phi-4/evil_neg_instruct.csv \
  --trait evil \
  --save_dir persona_vectors/phi-4/ \
  --threshold 50

