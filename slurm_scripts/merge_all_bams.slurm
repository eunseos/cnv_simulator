#!/bin/bash
#SBATCH --job-name=final_merge
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools

# Directory containing sorted batch BAM files
SORTED_DIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2/merged_bams_3/sampled_bams"
FINAL_MERGED_BAM="${SORTED_DIR}/final_merged.bam"
FINAL_SORTED_BAM="${SORTED_DIR}/final_merged_sorted.bam"

# Merge sorted batch BAMs (result is unsorted)
echo "Merging sorted batch BAM files..."
ls -lh ${SORTED_DIR}/merged_batch_*_sorted_sampled.bam
samtools merge -@ $SLURM_CPUS_PER_TASK -c -p --no-PG -o "$FINAL_MERGED_BAM" ${SORTED_DIR}/merged_batch_*_sorted_sampled.bam
# samtools merge -@ $SLURM_CPUS_PER_TASK -o "$FINAL_MERGED_BAM" ${SORTED_DIR}/merged_batch_0_sorted.bam ${SORTED_DIR}/merged_batch_1_sorted.bam

# Sort the merged BAM
echo "Sorting the merged BAM..."
samtools sort -@ $SLURM_CPUS_PER_TASK -o "$FINAL_SORTED_BAM" "$FINAL_MERGED_BAM"

# Index the sorted merged BAM
echo "Indexing the sorted merged BAM..."
samtools index "$FINAL_SORTED_BAM"

echo "Done! Final merged sorted BAM is: $FINAL_SORTED_BAM"