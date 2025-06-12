#!/bin/bash
#SBATCH --job-name=another_merge
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

BAM_DIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2/merged_bams"
OUTPUT_DIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2/final_merged_bam"
mkdir -p "$OUTPUT_DIR"

find "$BAM_DIR" -name '*.bam' | {
    read firstbam
    samtools view -h "$firstbam"
    while read bam; do
        samtools view "$bam"
    done
} | samtools view -ubS - | samtools sort -o "$OUTPUT_DIR/merged.bam"
samtools index "$OUTPUT_DIR/merged.bam"
ls -l "$OUTPUT_DIR/merged.bam" "$OUTPUT_DIR/merged.bam.bai"