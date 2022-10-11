#!/bin/bash
#SBATCH --job-name=MLM_OOD
#SBATCH --partition=t4v2

#SBATCH --gres=gpu:1

#SBATCH --qos=high

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=16G

#SBATCH --output=sentence_modi.out

#SBATCH --error=sentence_modi.out

# prepare your environment here
source /h/elau/Conv_BERT/BERT_MLM_env/bin/activate

ipython MLM-OOD_v2.py
