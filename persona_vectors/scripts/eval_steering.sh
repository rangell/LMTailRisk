gpu=${1:-0}
model="./ckpt/Qwen2.5-7B-Instruct/qwen-evil_misaligned_2"
trait="evil"
vector_path="persona_vectors/Qwen2.5-7B-Instruct/evil_response_avg_diff.pt"
coef=-1.5
layer=20
steering_type="response"
output_path="eval_persona_eval/$(basename $model)/${trait}_steer_${steering_type}_layer${layer}_coef${coef}.csv"



CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
    --model $model \
    --trait $trait \
    --output_path $output_path \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version eval \
    --steering_type $steering_type \
    --coef $coef \
    --vector_path $vector_path \
    --layer $layer