#!/bin/bash

module load samtools

BAM_LIST="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/PM_0510_EGFR_2/merged_bams_3/bam_lst.txt"

total=0

while IFS= read -r bam; do
    if [[ -f "$bam" ]]; then
        count=$(samtools view -c "$bam")
        echo -e "$count"
        total=$((total + count))
    else
        echo "Warning: BAM file not found: $bam" >&2
    fi
done < "$BAM_LIST"

echo -e "\nTotal reads across all BAMs: $total"