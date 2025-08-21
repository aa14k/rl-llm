CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_math.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --gamma 0.99999975 \
    --seed 42 \
    --machine_name "5gen-capacityblock4" \
    --sync_ref_model \
    --disable_dropout \
    --shuffle_dataset \
    --num_generations 5 \
    --gradient_accumulation_steps 5 


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_math.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 0.99999975 \
#     --seed 42 \
#     --machine_name "300rew-3gen-capacityblock4-2" \
#     --sync_ref_model \
#     --disable_dropout \
#     --num_generations 3 \
#     --gradient_accumulation_steps 3 \


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_math.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 1.0 \
#     --seed 42 \
#     --machine_name "300rew-3gen-capacityblock4-1" \
#     --sync_ref_model \
#     --disable_dropout \
#     --num_generations 3 \
#     --gradient_accumulation_steps 3 \


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_math.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 1.0 \
#     --seed 42 \
#     --machine_name "300rew-3gen-capacityblock4-2" \
#     --sync_ref_model \
#     --disable_dropout \
#     --num_generations 3 \
#     --gradient_accumulation_steps 3 \


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_math.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 1.0 \
#     --seed 42 \
#     --machine_name "300rew-3gen-capacityblock4-3" \
#     --sync_ref_model \
#     --disable_dropout \
#     --num_generations 3 \
#     --gradient_accumulation_steps 3 \



# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_math.py \
#     --model_name "/home/ubuntu/alex/verifiers/outputs/Qwen2.5-7B-Instruct-math-gamma0.99999975-seed42-300rew-3gen-capacityblock4/checkpoint-1714" \
#     --gamma 1.0 \
#     --seed 42 \
#     --machine_name "continue" \
#     --sync_ref_model \
#     --disable_dropout \
#     --shuffle_dataset \
#     --save_steps 400 \
#     --learning_rate


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_math.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 0.99999975 \
#     --seed 42 \
#     --machine_name "300rew-3gen-capacityblock4-11" \
#     --sync_ref_model \
#     --disable_dropout \
#     --num_generations 3 \
#     --gradient_accumulation_steps 3 \
#     --shuffle_dataset


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_math.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 1.0 \
#     --seed 42 \
#     --machine_name "300rew-3gen-capacityblock4-10" \
#     --sync_ref_model \
#     --disable_dropout \
#     --num_generations 3 \
#     --gradient_accumulation_steps 3 \
#     --shuffle_dataset



# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_math.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 1.0 \
#     --seed 42 \
#     --machine_name "300rew-3gen-capacityblock4" \
#     --sync_ref_model \
#     --disable_dropout \
#     --num_generations 3 \
#     --gradient_accumulation_steps 3 