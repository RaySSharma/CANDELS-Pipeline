#!/bin/bash
# Number of Nodes:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -p main
#SBATCH -J powderday
#SBATCH --requeue

# Run command
python --version
which python

python run_pipeline.py