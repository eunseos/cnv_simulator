#!/bin/bash
#SBATCH --job-name=batch_merge_array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=componc_cpu
#SBATCH --array=0-28               # adjust this to number of batches - 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

set -euo pipefail

module load samtools

BAMDIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2/PM_0510_EGFR_2_intermediate_bams"
TMPDIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2/tmp_merge_bams"
mkdir -p "$TMPDIR"

BATCH_SIZE=30
OUTPUT_BATCH_DIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2/merged_bams_3"
mkdir -p "$OUTPUT_BATCH_DIR"

# List all BAMs sorted
find "$BAMDIR" -maxdepth 1 -type f -name "*.bam" | sort > "$TMPDIR/bam_list.txt"

# Calculate batch start and end lines
start=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE + 1 ))
end=$(( start + BATCH_SIZE - 1 ))
total_lines=$(wc -l < "$TMPDIR/bam_list.txt")
if [ $end -gt $total_lines ]; then
    end=$total_lines
fi

# Extract the batch BAM list
sed -n "${start},${end}p" "$TMPDIR/bam_list.txt" > "$TMPDIR/batch_${SLURM_ARRAY_TASK_ID}.txt"

# Check if batch file is empty
if [ ! -s "$TMPDIR/batch_${SLURM_ARRAY_TASK_ID}.txt" ]; then
    echo "No BAMs to merge for batch ${SLURM_ARRAY_TASK_ID}, skipping."
    exit 0
fi

# Merge batch
batch_output="${OUTPUT_BATCH_DIR}/merged_batch_${SLURM_ARRAY_TASK_ID}.bam"
echo "Merging BAMs for batch ${SLURM_ARRAY_TASK_ID} into $batch_output..."
samtools merge -@ $SLURM_CPUS_PER_TASK --verbosity 8 --no-PG -c -p -b "$TMPDIR/batch_${SLURM_ARRAY_TASK_ID}.txt" "$batch_output"
if [ $? -ne 0 ]; then
    echo "Error: samtools merge failed for batch ${SLURM_ARRAY_TASK_ID}. Check input BAMs and log for details."
    exit 1
fi

echo "Batch ${SLURM_ARRAY_TASK_ID} done at $(date)"