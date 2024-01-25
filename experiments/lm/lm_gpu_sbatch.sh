#!/bin/bash
#
#SBATCH --job-name=ner
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=100


#module purge
#module load anaconda3
#conda activate tf
source activate accord

python -m experiments.lm.train --model_name roberta-large --model_type roberta --wandb_api_key API_KEY
