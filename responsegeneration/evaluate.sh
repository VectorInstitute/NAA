#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH -p rtx6000
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -c 4
#SBATCH --mem-per-cpu=32GB
#SBATCH --output=evaluate.out
#SBATCH --error=evaluate.out

echo Running on $(hostname)

# run evaluation
python ./evaluate.py \
    --output_name gpt2large_marco_epoch5_evalresults.json \
    --log_dir /ssd003/home/shirleyw/ConversationalAI/QAChatbot/gpt2_checkpoints/ \
    --checkpoint /ssd003/home/shirleyw/ConversationalAI/QAChatbot/gpt2_checkpoints/checkpoint_epoch5_step1701210.pth \
    --valid_dataset_path /ssd003/projects/aieng/conversational_ai/data/MSMARCO/marco_valid_tokenized.json \
    --dataset_type marco
