#!/bin/bash
#SBATCH --job-name=norm_index_bam
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org


# Load modules (if needed)
module load samtools  # Adjust version/module as needed

BAMDIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2"

bam_path="${BAMDIR}/PM_0510_EGFR_2_unnormalized_cnv.bam"
baseline_bam_path="${BAMDIR}/PM_0510_EGFR_2_baseline_cells.bam"

if [ ! -f "$bam_path" ]; then
    echo "Error: $bam_path not found!"
    exit 1
fi
if [ ! -f "$baseline_bam_path" ]; then
    echo "Error: $baseline_bam_path not found!"
    exit 1
fi

baseline_read_count=$(samtools view -c "$baseline_bam_path")
echo "Baseline read count: $baseline_read_count"

unnormalized_read_count=$(samtools view -c "$bam_path")
echo "Unnormalized read count: $unnormalized_read_count"

# Calculate the scaling factor
scaling_factor=$(awk -v baseline="$baseline_read_count" -v unnormalized="$unnormalized_read_count" 'BEGIN { printf "%.6f", baseline / unnormalized }')
echo "Scaling factor: $scaling_factor"
# Normalize the BAM file
output_bam_path="${BAMDIR}/PM_0510_EGFR_2_final_cnv.bam"
samtools view -@ $SLURM_CPUS_PER_TASK --subsample-seed 283 --subsample "$scaling_factor" -b -o "$output_bam_path" "$bam_path"

# Sort the normalized BAM file
sorted_bam_path="${BAMDIR}/PM_0510_EGFR_2_final_sorted_cnv.bam"
samtools sort -@ $SLURM_CPUS_PER_TASK -o "$sorted_bam_path" "$output_bam_path"

# Index the sorted BAM file
samtools index "$sorted_bam_path"
echo "Normalization, sorting, and indexing complete."