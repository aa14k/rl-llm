CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_gsm8k_llama.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --gamma 1.0 \
    --seed 42 \
    --machine_name "3gen-capacityblock8" \
    --sync_ref_model \
    --disable_dropout \
    --shuffle_dataset \
    --num_generations 3 \
    --gradient_accumulation_steps 3 


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_gsm8k_llama.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 1.0 \
#     --seed 41 \
#     --machine_name "3gen-capacityblock8" \
#     --sync_ref_model \
#     --disable_dropout \
#     --shuffle_dataset \
#     --num_generations 3 \
#     --gradient_accumulation_steps 3 


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_gsm8k_llama.py \
#     --model_name "meta" \
#     --gamma 1.0 \
#     --seed 40 \
#     --machine_name "3gen-capacityblock8" \
#     --sync_ref_model \
#     --disable_dropout \
#     --shuffle_dataset \
#     --num_generations 3 \
#     --gradient_accumulation_steps 3 