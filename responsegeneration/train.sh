#!/bin/bash
#SBATCH --job-name=marco_gpt2
#SBATCH -p rtx6000
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -c 4
#SBATCH --mem-per-cpu=32GB
#SBATCH --output=gpt2finetune-v1.out
#SBATCH --error=gpt2finetune-v1.out

echo Running on $(hostname)

export LOCAL_RANK=-1

# run training
python ./finetune_gpt.py \
    --log_dir $PWD/gpt2_checkpoints/ \
    --model_type gpt2-large \
    --train_batch_size 1 \
    --valid_batch_size 5 \
    --train_dataset_path /ssd003/projects/aieng/conversational_ai/data/MSMARCO/marco_train_tokenized.json \
    --valid_dataset_path /ssd003/projects/aieng/conversational_ai/data/MSMARCO/marco_valid_tokenized.json \
    --max_len 330 \
    --n_epochs 5 \
    --random_seed 39 \
    --fp16 \
    --eval_before_start
