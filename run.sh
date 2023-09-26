#!/bin/bash

## begin SLURM Batch Commands
#SBATCH --output=output.log
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=7-10:00

module load python/3.10
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py --train --train_path /home/ndil/ChEMU2023_FOR_CLASS/train/ --dev_path /home/ndil/ChEMU2023_FOR_CLASS/dev/ --epochs 100
