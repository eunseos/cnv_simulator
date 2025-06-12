#!/bin/bash
#SBATCH --job-name=sample_normalize
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

set -euo pipefail

module load samtools

INPUT_DIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2/merged_bams_3/sorted_bams"       # Change this
OUTPUT_DIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2/merged_bams_3/sampled_bams"   # Change this
TARGET_TOTAL_READS=60184306

mkdir -p "$OUTPUT_DIR"

declare -A read_counts
total_reads=0

echo "Counting reads in each BAM..."

# Step 1: Count reads
for bam in "$INPUT_DIR"/*.bam; do
    if [[ -f "$bam" ]]; then
        count=$(samtools view -c "$bam")
        read_counts["$bam"]=$count
        total_reads=$((total_reads + count))
    else
        echo "Warning: BAM file not found: $bam" >&2
    fi
done

echo "Total reads across all BAMs: $total_reads"
echo

# Step 2: Sample reads proportionally
for bam in "${!read_counts[@]}"; do
    count=${read_counts[$bam]}

    # Fraction of total, rounded to 10 decimal places
    fraction=$(awk -v c="$count" -v t="$total_reads" 'BEGIN { printf "%.10f", c/t }')

    # Target reads (as integer), then recalculate sample fraction to get accurate final sample
    target_reads=$(awk -v f="$fraction" -v tr="$TARGET_TOTAL_READS" 'BEGIN { printf "%.0f", f * tr }')
    echo "Target reads for $bam: $target_reads"
    sample_fraction=$(awk -v tr="$target_reads" -v c="$count" 'BEGIN { printf "%.3f", tr/c }')
    echo "Sample fraction for $bam: $sample_fraction"

    bam_basename=$(basename "$bam" .bam)
    echo "Processing BAM: $bam_basename"
    output_bam="$OUTPUT_DIR/${bam_basename}_sampled.bam"
    echo "Output BAM: $output_bam"

    echo "Sampling $target_reads reads from $bam (sample_fraction: $sample_fraction)"
    samtools view -@ 4 --subsample-seed 283 --subsample "$sample_fraction" -b -o "$output_bam" "$bam"
done

echo
echo "Done. Sampled BAMs saved to $OUTPUT_DIR"