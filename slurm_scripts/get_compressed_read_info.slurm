#!/bin/bash
#SBATCH --job-name=get_compressed_read_info
#SBATCH --output=logs/%x_$A_$a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --partition=componc_cpu
#SBATCH --array=0-14
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools

BASEDIR="/data1/shahs3/users/sunge/cnv_simulator"
BAMDIR="${BASEDIR}/data/normal_cell_bams"

BAM_FILES=($BAMDIR/*.bam)

BAM_FILE=${BAM_FILES[$SLURM_ARRAY_TASK_ID]}
BASE_NAME=$(basename "$BAM_FILE" .bam)
PARQUET_FILE="${BAMDIR}/${BASE_NAME}_compressed_read_info.parquet"

echo "Processing BAM file: $BAM_FILE"
echo "Outputting to: $PARQUET_FILE"

# Check if the output file already exists
if [[ -f "$PARQUET_FILE" ]]; then
    echo "Output file already exists. Skipping."
    exit 0
fi

# Extract compressed read information
samtools view -@ 16 "$BAM_FILE" | \
awk '{print $1, $3, $4, $8, $12}' > tmp_read_info.txt

# Convert to Parquet format
python3 -c "
import pandas as pd
data = pd.read_csv('tmp_read_info.txt', sep=' ', header=None,
                 names=['QNAME', 'RNAME', 'POS', 'MAPQ', 'CIGAR'])
