#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --mail-type=fail
#SBATCH --output ./{jobname}.%J.out
#SBATCH --chdir=./
#SBATCH -N 1
{requirements}

set -e

export OMP_NUM_THREADS={cores_per_task}
export MPLCONFIGDIR="${{TMPDIR:-/tmp}}/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"
{environment_setup}

echo "Running task {task_id} specified in {config_file} on $HOSTNAME at $(date)"
{python_executable} {cluster_jobs_module} run {config_file} {task_id} &> "{jobname}.task_{task_id}.out"
echo "finished at $(date)"
