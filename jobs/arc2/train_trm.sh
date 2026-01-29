#!/bin/bash -l
#SBATCH --job-name=trm_arc2
#SBATCH --output=job_logs/%x_%j.log
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --time=08:00:00
#SBATCH --nodes=1

cd $HOME/recursive_models
source $HOME/recursive_models/scripts/activate.sh
set -euo pipefail



python -m src.train   vocab=arc2   model=trm
