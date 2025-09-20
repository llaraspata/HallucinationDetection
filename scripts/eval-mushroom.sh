#!/bin/bash

#SBATCH -A IscrC_EXAM
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=24:00:00
#SBATCH -N 1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=10000
#SBATCH --job-name=eval-mushroom
#SBATCH --out=output.log
#SBATCH --err=error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it

export WANDB_MODE="offline"

source .venv/bin/activate

# Delete previous wandb offline runs
rm -rf wandb/

srun -u python -W ignore -m src.evaluation.eval --model_name "meta-llama/Meta-Llama-3-8B" --data_name "mushroom"