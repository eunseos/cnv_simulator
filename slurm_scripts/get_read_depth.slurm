#!/bin/bash
#SBATCH --job-name=read_depth
#SBATCH --output=logs/read_depth.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=5:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load miniforge3
conda init bash
source ~/.bashrc
conda activate seqtools_env

BASEDIR="/data1/shahs3/users/sunge/cnv_simulator"
BAMDIR="${BASEDIR}/synthetic_bams"
OUTDIR="${BASEDIR}/synthetic_bams/read_depth"
SAMPLE_NAME="minitest_c3_8_sg0_500cells_cnv"

profile_path="${BASEDIR}/data/small_cnv_profiles/${SAMPLE_NAME}_profile.tsv"

echo ${OUTDIR}/${SAMPLE_NAME}
echo ${BAMDIR}/${SAMPLE_NAME}.bam
echo ${profile_path}

mosdepth --no-per-base --fast-mode --threads 32 --by 10000 ${OUTDIR}/${SAMPLE_NAME} ${BAMDIR}/${SAMPLE_NAME}.bam