#!/bin/bash

#SBATCH -A IscrC_EXAM
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=24:00:00
#SBATCH -N 1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=4
#SBATCH --mem=123000
#SBATCH --job-name=predict-snyder
#SBATCH --out=output.log
#SBATCH --err=error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it

source .venv/bin/activate

srun -u python -W ignore -m src.model.predict_snyder_et_al --model_name "falcon-7b" --data_name "mushroom" --use_local

srun -u python -W ignore -m src.model.predict_snyder_et_al --model_name "falcon-7b" --data_name "halu_eval" --use_local

srun -u python -W ignore -m src.model.predict_snyder_et_al --model_name "falcon-7b" --data_name "halu_bench" --use_local
