#!/usr/bin/env bash

#SBATCH -o stdout-%a.txt
#SBATCH -e stderr-%a.txt
#SBATCH -p atlas_all -c 2 --mem 3000
#SBATCH -a 0-35
#SBATCH -t 01:00:00

set -eu

echo "submit from ${SLURM_SUBMIT_DIR-here}, index ${SLURM_ARRAY_TASK_ID-none}"
cd ${SLURM_SUBMIT_DIR-.}

FN=${SLURM_ARRAY_TASK_ID-0}

calculate_auc_numpy.py dan hl_tracks --file-number $FN -o hl_tracks
calculate_auc_numpy.py dan tracks --file-number $FN -o tracks

echo "done"
