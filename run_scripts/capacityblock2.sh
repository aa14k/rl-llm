python grpo_discounted.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --gamma 0.99999975 \
    --seed 43 \
    --machine_name "5gen-1epoch-capacityblock2"\
    --sync_ref_model \
    --disable_dropout \
    --ref_model_sync_steps 16 \
    --warmup_ratio 0.1 \
    --num_generations 5 \
    --gradient_accumulation_steps 5 \


# python grpo_discounted.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 1.0 \
#     --seed 40 \
#     --machine_name "constantlr-1epoch-capacityblock2" \
#     --sync_ref_model \
#     --disable_dropout \
#     --ref_model_sync_steps 16 \

# python grpo_discounted.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 1.0 \
#     --seed 39 \
#     --machine_name "constantlr-1epoch-capacityblock2" \
#     --sync_ref_model \
#     --disable_dropout \
#     --ref_model_sync_steps 16 \

# python grpo_discounted.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --gamma 1.0 \
#     --seed 38 \
#     --machine_name "constantlr-1epoch-capacityblock2" \
#     --sync_ref_model \
#     --disable_dropout \
#     --ref_model_sync_steps 16 \