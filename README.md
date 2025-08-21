# Setup

To get everything setup, simply run the commands inside 

setup.sh

This should install a conda environment and download the necessary dependencies. 

Note the scripts are meant to work with vLLM (which assumes you have at least 2 gpus). If this is not the case, please go into either train_gsm8k.py or train_math.py and set use_vllm = False in the GRPOConfig arguments. Also comment out the generation_kwargs beneath it. 

start vllm by running the following command

CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --data-parallel-size 1

then you can run the script by running

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 --config_file configs/ddp.yaml accelerate launch train_gsm8k.py


