#!/bin/bash
#SBATCH --job-name=get_bed_coverage
#SBATCH --output=logs/get_bed_coverage_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --partition=componc_cpu
#SBATCH --array=0-12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools
source activate seqtools_env

# Define the group list (group names as strings "0" through "12")
GROUP_LIST=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")
GROUP=${GROUP_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Processing group: $GROUP"

# Set paths
DATADIR="/data1/shahs3/users/sunge/cnv_simulator/data"
BAMDIR="${DATADIR}/normal_cell_bams"              # contains BAM files
BEDDIR="${DATADIR}/normal_cell_beds"               # contains BED files

# Set vars
bin_size=100000
bin_path="${DATADIR}/genome_${bin_size}_bins.bed"
group_bam="${BAMDIR}/group_${GROUP}_merged.bam"

# Check if the group BAM file exists
if [[ ! -f "$group_bam" ]]; then
    echo "Warning: Group BAM file not found for group $GROUP at $group_bam" >&2
    exit 1
fi

# Run bedtools coverage
bedtools coverage -a $bin_path -b $group_bam > ${BEDDIR}/group_${GROUP}_${bin_size}_coverage.bed