#!/bin/bash
# Number of Nodes:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --array=1-43
#SBATCH -t 2:00:00
#SBATCH -p main
#SBATCH -J AGN-Mergers
#SBATCH --requeue

# Set per_task = num_images / num_array_jobs
PER_TASK=87
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

# Run command
PYTHON=/home/rss230/anaconda2/envs/CANDELS-pipeline/bin/python
$PYTHON --version
which $PYTHON

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# Run the loop of runs for this task.
for (( run=$START_NUM; run<=END_NUM; run++ )); do
  echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
  image=$(ls /scratch/rss230/sharma_choi_images/realsim_images_071321/*.fits | sed -n ${run}p)
  $PYTHON run_pipeline.py $image
done

date
