# # Step 1a: Generate impressions in a single job
# # python generate_f2i_responses.py \
# #     --split test \
# #     --output results/medgemma_f2i.jsonl \
# #     --num_samples 10000 \
# #     --batch_size 64 \
# #     --max_new_tokens 512 \
# #     --backend vllm \
# #     --tensor_parallel_size 1 \
# #     --max_model_len 4096

# # # Step 1b: Generate impressions in parallel chunks (10 jobs → results/medgemma_f2i_job000X_of0010.jsonl)
# # NUM_JOBS=10
# # for JOB_INDEX in $(seq 0 $((NUM_JOBS - 1))); do
# #     python generate_f2i_responses.py \
# #         --split test \
# #         --num_samples 10000 \
# #         --batch_size 64 \
# #         --max_new_tokens 512 \
# #         # --backend vllm \
# #         # --tensor_parallel_size 1 \
# #         # --max_model_len 4096 \
# #         --job_index "$JOB_INDEX" \
# #         --num_jobs "$NUM_JOBS" &
# # done
# # wait
# # echo "All generation jobs done."

# # Step 1b: Generate impressions in parallel chunks (10 jobs → results/medgemma_f2i_job000X_of0010.jsonl)

# NUM_JOBS=10
# python generate_f2i_responses.py \
#     --split test \
#     --batch_size 64 \
#     --max_new_tokens 512 \
#     --job_index 0 \
#     --num_jobs "$NUM_JOBS" 

# # Step 2a: Evaluate in a single job
# # python eval_crimson.py \
# #     --input results/medgemma_f2i.jsonl \
# #     --output results/crimson_scores.json \
# #     --batch-size 8 \
# #     --details


# python eval_crimson.py \
#     --input results/medgemma_f2i.jsonl \
#     --batch-size 8 \
#     --details \
#     --job-index 0 \
#     --num-jobs "$NUM_JOBS" 

# # # Step 2b: Evaluate in parallel chunks (e.g. 10 jobs over 10k samples → 1k each)
# # NUM_JOBS=10
# # for JOB_INDEX in $(seq 0 $((NUM_JOBS - 1))); do
# #     python eval_crimson.py \
# #         --input results/medgemma_f2i.jsonl \
# #         --batch-size 8 \
# #         --details \
# #         --job-index "$JOB_INDEX" \
# #         --num-jobs "$NUM_JOBS" &
# # done
# # wait
# # echo "All jobs done."


JOB_ID=0
NUM_JOBS=4

# TESTNAME="results/test_medgemma_f2i_${JOB_ID}.jsonl"
VALNAME="results/scratch_val_medgemma_f2i_${JOB_ID}.jsonl"
# TRAINNAME="results/train_medgemma_f2i_${JOB_ID}.jsonl"

# generate
# python generate_f2i_responses.py --split test --output $TESTNAME --batch_size 64 --max_new_tokens 512 --job_index  $SLURM_ARRAY_TASK_ID  --num_jobs $NUM_JOBS
python generate_f2i_responses.py --split validation --output $VALNAME --batch_size 64 --max_new_tokens 512 --job_index  $JOB_ID  --num_jobs $NUM_JOBS
# python generate_f2i_responses.py --split train --output $TRAINNAME --batch_size 64 --max_new_tokens 512 --job_index  $SLURM_ARRAY_TASK_ID  --num_jobs $NUM_JOBS



# TESTEVALNAME="results/crimson_test_medgemma_f2i_${JOB_ID}.jsonl"
VALEVALNAME="results/scratch_crimson_val_medgemma_f2i_${JOB_ID}.jsonl"
# TRAINEVALNAME="results/crimson_train_medgemma_f2i_${JOB_ID}.jsonl"

# eval
# python eval_crimson.py  --input $TESTNAME --output $TESTEVALNAME  --batch-size 32  --details --job-index  $SLURM_ARRAY_TASK_ID --num-jobs $NUM_JOBS
python eval_crimson.py  --input $VALNAME --output $VALEVALNAME  --batch-size 32  --details --job-index  0 --num-jobs 1
# python eval_crimson.py  --input $TRAINNAME --output $TRAINEVALNAME   --batch-size 32  --details --job-index  $SLURM_ARRAY_TASK_ID --num-jobs $NUM_JOBS
