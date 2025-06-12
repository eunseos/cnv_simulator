#!/bin/bash
#SBATCH --job-name=sort_bam
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

# Load samtools module if necessary (depends on your system)
module load samtools

# Set paths
OUT_DIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2/merged_bams_3"
BAM_DIR="${OUT_DIR}/sampled_bams"
SORTED_DIR="${OUT_DIR}/sampled_sorted_bams"
MERGED_BAM="${OUT_DIR}/decompressed_merge.bam"
FINAL_BAM="${OUT_DIR}/merged_output_final.bam"

# Make output directory for sorted files
# mkdir -p "$SORTED_DIR"

# 1. Sort each BAM file with fast compression (-l 1)
echo "Sorting BAM files..."

bam_files=("$BAM_DIR"/*.bam)
if [ ${#bam_files[@]} -eq 0 ]; then
    echo "No BAM files found in $BAM_DIR"
    exit 1
fi

# for bam in "$BAM_DIR"/*.bam; do
#     echo "Sorting $bam..."
#     base=$(basename "$bam" .bam)
#     samtools sort -@ $SLURM_CPUS_PER_TASK -l 1 -o "$SORTED_DIR/${base}_sorted.bam" "$bam"
# done


# 2. Merge sorted BAMs with no compression (-l 0)
echo "Merging sorted BAMs..."
samtools merge -@ $SLURM_CPUS_PER_TASK -c -p --no-PG -l 0 "$MERGED_BAM" "$SORTED_DIR"/*_sorted.bam

# 3. Optional: Recompress the merged BAM with higher compression (e.g. -l 5)
echo "Recompressing merged BAM..."
samtools sort -@ $SLURM_CPUS_PER_TASK -l 5 -o "$FINAL_BAM" "$MERGED_BAM"

# 4. Index the final BAM
echo "Indexing the final BAM..."
samtools index "$FINAL_BAM"

echo "Done!"

