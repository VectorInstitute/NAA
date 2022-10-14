#!/bin/bash
#SBATCH --job-name=class_intents
#SBATCH --partition=t4v2

#SBATCH --gres=gpu:1

#SBATCH --qos=high

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=16G

#SBATCH --output=class_intents_simplified_banking.out

#SBATCH --error=class_intents_simplified_banking.out

# prepare your environment here


source /h/elau/Conv_BERT/BERT_MLM_env/bin/activate

ipython banking_text_classification_with_BERT_intents.py
