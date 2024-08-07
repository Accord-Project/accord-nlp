#!/bin/bash
#
#SBATCH --job-name=ner_cv
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=100


#module purge
#module load anaconda3
#conda activate tf
source activate accord

#srun python3 experiments/ner/ner_cv_experiments.py --model_name bert-base-cased --model_type bert --k_folds 2
python -m experiments.ner.ner_cv_experiments --model_name bert-base-cased --model_type bert --k_folds 5 --wandb_api_key API_KEY
