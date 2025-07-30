#!/bin/bash

#SBATCH -A [removed for anonymization]
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=24:00:00
#SBATCH -N 1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=10000
#SBATCH --job-name=predict-mushroom
#SBATCH --out=output.log
#SBATCH --err=error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@domain.com

source .venv/bin/activate

srun -u python -W ignore -m src.model.predict --model_name "meta-llama/Meta-Llama-3-8B" --data_name "mushroom" --use_local