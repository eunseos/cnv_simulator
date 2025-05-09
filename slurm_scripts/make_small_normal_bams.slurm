#!/bin/bash
#SBATCH --job-name=filter_small_normal_bams
#SBATCH --output=logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools

# Set paths
small_sample_name="sg0_500cells"

DATADIR="/data1/shahs3/users/sunge/cnv_simulator/data"
CSV_PATH="${DATADIR}/${small_sample_name}.csv"
output_path="${DATADIR}/normal_cell_bams/${small_sample_name}.bam"
source_bam_path="${DATADIR}/normal_cell_bams/group_0_merged.bam"

BARCODE_LIST=$(mktemp)
tail -n +2 "$CSV_PATH" | cut -d ',' -f 14 | tr -d '"' > "$BARCODE_LIST"

barcode_filters=$(awk '{print "-d CB:"$1}' "$BARCODE_LIST" | tr '\n' ' ')

if [[ -z "$barcode_filters" ]]; then
    echo "No barcodes found for $small_sample_name" >&2
    exit 1
fi

read -ra BARCODE_ARGS <<< "$barcode_filters"

echo "Processing sample: $small_sample_name"
samtools view -@ 32 -b "${BARCODE_ARGS[@]}" "$source_bam_path" 1 2 -o "$output_path"

echo "Indexing BAM file for $small_sample_name"
samtools index -@ 32 "$output_path"

rm "$BARCODE_LIST"