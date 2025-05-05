#!/bin/bash
#SBATCH --job-name=filter_by_group
#SBATCH --output=logs/filter_by_group_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --partition=componc_cpu
#SBATCH --array=0-12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools

# Define the group list (group names as strings "0" through "12")
GROUP_LIST=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")
GROUP=${GROUP_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Processing group: $GROUP"

# Set paths
DATADIR="/data1/shahs3/users/sunge/cnv_simulator/data"
CSV_PATH="${DATADIR}/all_normal_samples.csv"              # replace with actual CSV path
BARCODE_DIR="${DATADIR}/normal_cell_barcodes"              # contains barcode .txt files
OUTPUT_DIR="${DATADIR}/normal_cell_bams"               # where to save grouped BAMs
mkdir -p "$OUTPUT_DIR"

# Temp list of BAM files to merge
MERGE_LIST=$(mktemp)

trap 'rm -f "$MERGE_LIST"' EXIT

# Process CSV entries
tail -n +2 "$CSV_PATH" | while IFS=',' read -r _ bam_path sample_name _ _ group _; do
    group=$(echo "$group" | tr -d '"')
    bam_path=$(echo "$bam_path" | tr -d '"')
    sample_name=$(echo "$sample_name" | tr -d '"')

    if [[ "$group" != "$GROUP" ]]; then continue; fi

    barcode_file="${BARCODE_DIR}/${sample_name}_cell_barcodes.txt"
    out_bam="${OUTPUT_DIR}/${sample_name}_filtered.bam"

    echo "Processing sample: $sample_name"

    if [[ ! -f "$bam_path" ]]; then
        echo "Warning: BAM not found for $sample_name at $bam_path" >&2
        continue
    fi

    if [[ ! -s "$barcode_file" ]]; then
        echo "Warning: Barcode file missing or empty for $sample_name at $barcode_file" >&2
        continue
    fi

    # Build barcode filter string
    barcode_filters=$(awk '{print "-d CB:"$1}' "$barcode_file" | tr '\n' ' ')

    # Skip if no barcodes found
    if [[ -z "$barcode_filters" ]]; then
        echo "No barcodes found for $sample_name. Skipping."
        continue
    fi

    samtools view -@ 32 -b $barcode_filters "$bam_path" -o "$out_bam"
    echo "$out_bam" >> "$MERGE_LIST"
    echo "Saved filtered BAM: $out_bam"
done

# Merge BAMs for the group
MERGED_BAM="${OUTPUT_DIR}/group_${GROUP}_merged.bam"
samtools merge -@ 32 "$MERGED_BAM" $(cat "$MERGE_LIST")
samtools index -@ 32 "$MERGED_BAM"

# Cleanup
xargs rm -f < "$MERGE_LIST"
rm "$MERGE_LIST"
echo "Finished processing group: $GROUP"