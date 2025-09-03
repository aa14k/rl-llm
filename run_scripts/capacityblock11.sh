
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_big.py \
    --seed 42 \
    --machine_name "big-capacityblock11" \
    --sync_ref_model \
    --disable_dropout \
    --shuffle_dataset \
    --gamma 0.999