#!/bin/bash
#SBATCH --job-name=WISDM	      # job name, Job.sh
#SBATCH --cpus-per-task=16      # job cpu request
#SBATCH --mem=64gb						  # job memory request
#SBATCH --time=01:00:00				  # time limit hh:mm:ss
#SBATCH --nodelist=fang-s009    # for GPU
#SBATCH --output=WISDM_%j.log   # standard output log


pwd; hostname; date

echo "Running Job"

source /home/student17/venv/bin/activate
python3 train_tf.py
deactivate

echo
date
