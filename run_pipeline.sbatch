#!/bin/bash
# Number of Nodes:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 05:00:00
#SBATCH -p main
#SBATCH -J powderday

# Run command
python --version
which python

python run_pipeline.py