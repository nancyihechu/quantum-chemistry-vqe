#!/bin/bash
# Run any pipeline script on a compute node.
# Usage: sbatch scripts/slurm_job.sh scripts/run_sqd_sim.py n2_631g_8o --noisy
#SBATCH --partition=general
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/%x-%j.out

set -euo pipefail
source /cluster/pixstor/dkhf3-lab/users/nancy/envs/qchem-sqd/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python "$@"
