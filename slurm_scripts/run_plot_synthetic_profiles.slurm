#!/bin/bash
#SBATCH --job-name=plot_synthetic_profiles
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

conda init bash
source ~/.bashrc
conda activate scanpy_env
echo "Active Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

test_name=$1

BASEDIR="/data1/shahs3/users/sunge/cnv_simulator"

python ${BASEDIR}/python_scripts/plot_synthetic_profiles.py --test_name $test_name