#!/bin/bash
#SBATCH --job-name=sample_cnv_reads_minitest_c2_5
#SBATCH --output=logs/%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools
module load miniforge3
conda init bash
source ~/.bashrc
conda activate scanpy_env
echo "Active Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Print current date and time
echo "Current date and time: $(date)"

profile_name="minitest_c2_5"

python /data1/shahs3/users/sunge/cnv_simulator/python_scripts/sample_cnv_reads.py --profile_name $profile_name

# Print current date and time
echo "Current date and time: $(date)"