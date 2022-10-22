#!/bin/bash
#SBATCH --job-name=distmarco_dialogpt
#SBATCH -p rtx6000
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH -c 4
#SBATCH --mem-per-cpu=32GB
#SBATCH --output=distdialofinetune.out
#SBATCH --error=distdialofinetune.out
#SBATCH --open-mode=append

export NODE_RANK=0

echo Running on $(hostname)

# run training
python -m torch.distributed.run \
    --master_addr="localhost" \
    --master_port=12345 \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=$NODE_RANK \
    ./finetune_gpt2_qa.py \
        --checkpoint_dir $PWD/distdialo_checkpoints/ \
        --model_type microsoft/DialoGPT-large \
        --train_batch_size 1 \
        --valid_batch_size 5 \
        --train_dataset_path data/MSMARCO/marco_train_tokenized.json \
        --valid_dataset_path data/MSMARCO/marco_valid_tokenized.json \
        --max_len 300 \
        --eval_before_start \
        --num_gpus 4
