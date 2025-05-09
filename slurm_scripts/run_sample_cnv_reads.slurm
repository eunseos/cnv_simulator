#!/bin/bash
#SBATCH --job-name=sample_cnv_reads
#SBATCH --output=logs/sample_cnv_reads_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools
module load miniforge3
conda init bash
source ~/.bashrc
conda activate scanpy_env
echo "Active Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

python /data1/shahs3/users/sunge/cnv_simulator/python_scripts/sample_cnv_reads.py