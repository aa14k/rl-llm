accelerate launch --config_file configs/zero3.yaml train_grpo_discounted_math.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --gamma 0.9999999 \
    --seed 42 \
    --machine_name "math-228" \
    --disable_dropout \
    --num_generations 4 \
    --gradient_accumulation_steps 4 \