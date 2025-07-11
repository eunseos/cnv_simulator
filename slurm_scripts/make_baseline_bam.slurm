#!/bin/bash
#SBATCH --job-name=baseline_bam
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools

# Print current date and time
echo "Current date and time: $(date)"

PROFILE_NAME=$1
GROUP_NAME=$2
if [ -z "$PROFILE_NAME" ]; then
    echo "Error: No profile name provided."
    exit 1
fi
echo "Profile name: $PROFILE_NAME"
echo "Group name: $GROUP_NAME"

# Define variables
BASEDIR="/data1/shahs3/users/sunge/cnv_simulator"
GROUP_BAM_DIR="${BASEDIR}/data/normal_cell_bams"
PROFILE_DIR="${BASEDIR}/synthetic_bams_2/${PROFILE_NAME}"


if [ -z "$GROUP_NAME" ]; then
    PROFILE_PATH="${PROFILE_DIR}/${PROFILE_NAME}_cell_profile.tsv"
else
    PROFILE_PATH="${PROFILE_DIR}/${PROFILE_NAME}_${GROUP_NAME}_cell_profile.tsv"
fi
RAW_BAM_PATH="${PROFILE_DIR}/${PROFILE_NAME}_sorted.bam"
OUT_BAM_PATH="${PROFILE_DIR}/${PROFILE_NAME}_baseline_cells.bam"

# Check if output BAM file already exists
if [[ -f "$OUT_BAM_PATH" ]]; then
    echo "Output BAM file already exists: $OUT_BAM_PATH"
    exit 0
fi

# Create output directory if it doesn't exist
mkdir -p "$PROFILE_DIR"

echo "Reading profile: $PROFILE_PATH"

# Extract baseline cells and groups
BASELINE_GROUPS_UNIQUE=$(awk -F'\t' '$1 == -1 {gsub(/"/, "", $NF); print $NF}' "$PROFILE_PATH" | tr ',' '\n' | sort -u)
ALL_CHROMOSOMES=$(awk -F'\t' 'NR > 2 && $2 != 0 {print $2}' "$PROFILE_PATH" | sort -u)

echo "Chromosomes: $ALL_CHROMOSOMES"
echo "Baseline groups (unique): $BASELINE_GROUPS_UNIQUE"

# Temp list of BAM files to merge
MERGE_LIST=$(mktemp)
trap 'rm -f "$MERGE_LIST"' EXIT

for group in $BASELINE_GROUPS_UNIQUE; do
    if [[ "$group" == "$GROUP_NAME" ]]; then
        BAM_PATH="${GROUP_BAM_DIR}/${group}.bam"
    else
        BAM_PATH="${GROUP_BAM_DIR}/group_${group}_merged.bam"
    fi

    if [[ ! -f "$BAM_PATH" ]]; then
        echo "Error: BAM file for group $group does not exist at $BAM_PATH" >&2
        exit 1
    fi

    echo "Processing group $group..."

    # Get barcodes for this group
    GROUP_BARCODES=$(awk -F'\t' -v group="$group" '$1 == -1 {
        gsub(/"/, "", $(NF-1));
        gsub(/"/, "", $NF);
        split($(NF-1), cells, ",");
        split($NF, groups, ",");
        for (i = 1; i <= length(groups); i++) {
            if (groups[i] == group) {
                print cells[i]
            }
        }
    }' "$PROFILE_PATH")

    if [[ -z "$GROUP_BARCODES" ]]; then
        echo "No barcodes for group $group, skipping."
        continue
    fi

    echo "Number of barcodes for group $group: $(echo "$GROUP_BARCODES" | wc -l)"

    # Create barcode filter
    BARCODE_FILTERS=$(echo "$GROUP_BARCODES" | awk '{print "-d CB:"$1}' | tr '\n' ' ')

    # Define temporary BAM file for this group
    OUT_BAM_TMP="${PROFILE_DIR}/${PROFILE_NAME}_${group}_baseline_cells_tmp.bam"

    # Write the temporary BAM file with select chromosomes and cells to the merge list
    read -ra BARCODE_ARGS <<< "$BARCODE_FILTERS"
    readarray -t CHROM_ARGS <<< "$ALL_CHROMOSOMES"
    samtools view -@ $SLURM_CPUS_PER_TASK -b "${BARCODE_ARGS[@]}" "$BAM_PATH" "${CHROM_ARGS[@]}" -o "$OUT_BAM_TMP"
    
    echo "$OUT_BAM_TMP" >> "$MERGE_LIST"
done

# Merge all temporary BAM files into the final baseline BAM
echo "Merging BAM files into: $RAW_BAM_PATH"
samtools merge -@ $SLURM_CPUS_PER_TASK -f "$RAW_BAM_PATH" $(cat "$MERGE_LIST")
samtools sort -@ $SLURM_CPUS_PER_TASK -o "$OUT_BAM_PATH" "$RAW_BAM_PATH"
samtools index -@ $SLURM_CPUS_PER_TASK "$OUT_BAM_PATH"

xargs rm -f < "$MERGE_LIST"

rm $RAW_BAM_PATH

echo "Baseline BAM file created at: $OUT_BAM_PATH"

echo "Current date and time: $(date)"