#!/bin/bash
#SBATCH --job-name=compute_read_depths
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools
module load miniforge3
conda init bash
source ~/.bashrc
conda activate scanpy_env

# Print current date and time
echo "Current date and time: $(date)"

profile_name=$1
if [ -z "$profile_name" ]; then
    echo "Error: No profile name provided."
    exit 1
fi
echo "Profile name: $profile_name"

BASEDIR="/data1/shahs3/users/sunge/cnv_simulator"
PROFILE_DIR="${BASEDIR}/synthetic_bams_2/${profile_name}"

# Create output directory if it doesn't exist
mkdir -p "$PROFILE_DIR"
mkdir -p "${PROFILE_DIR}/logs"

bam_filename="${profile_name}_final_sorted_cnv.bam"
if [ ! -f "${PROFILE_DIR}/$bam_filename" ]; then
    echo "Error: BAM file not found: $bam_filename"
    exit 1
fi
echo "BAM file: $bam_filename"

bin_size=500000
gc_wig_path="${BASEDIR}/data/refs_2/GRCh37-lite.fa.gc_500000_forgiving.wig"
gene_windows_path="/data1/shahs3/users/sunge/cnv_simulator/data/genome_5kb_bins.bed"

log_dir="${PROFILE_DIR}/logs"
mkdir -p "$log_dir"
log_path="${log_dir}/${profile_name}_compute_read_depths_hmmcopy_$(date +'%Y%m%d_%H%M%S').log"

python ${BASEDIR}/python_scripts/compute_read_depths.py \
    --profile_dir $PROFILE_DIR \
    --profile_name $profile_name \
    --bam_filename $bam_filename \
    --bin_size $bin_size \
    --gc_wig $gc_wig_path \
    --verbose \
    > "$log_path" 2>&1

echo "Read depth computation completed. Log file: $log_path"

log_path="${log_dir}/${profile_name}_compute_read_depths_5000_$(date +'%Y%m%d_%H%M%S').log"

python ${BASEDIR}/python_scripts/compute_read_depths.py \
    --profile_dir $PROFILE_DIR \
    --profile_name $profile_name \
    --bam_filename $bam_filename \
    --bin_size 5000 \
    --gene_windows $gene_windows_path \
    --verbose \
    > "$log_path" 2>&1

echo "Gene window read depth computation completed. Log file: $log_path"