#!/bin/bash
#SBATCH --job-name=read_depth
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${2}
#SBATCH --mem=128G
#SBATCH --time=5:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools
module load miniforge3
conda init bash
source ~/.bashrc
conda activate seqtools_env

BASEDIR="/data1/shahs3/users/sunge/cnv_simulator"
BAMDIR="${BASEDIR}/synthetic_bams"
OUTDIR="${BASEDIR}/synthetic_bams/read_depth"
SAMPLE_NAME=$1
NCORES=$2

bam_path="${BAMDIR}/${SAMPLE_NAME}_final_sorted_cnv.bam"
barcodes_path="${BAMDIR}/${SAMPLE_NAME}_final_sorted_cnv.bam.barcodes.txt"

TMP_BAMDIR="${BAMDIR}/tmp"
mkdir -p "$TMP_BAMDIR"

# Extract barcodes from the BAM file


# Get read depth for each barcode
# Write read depth to a temporary file

# Merge read depth temporary files into final output file, should be compressed
