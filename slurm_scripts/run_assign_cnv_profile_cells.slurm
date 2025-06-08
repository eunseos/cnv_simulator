#!/bin/bash
#SBATCH --job-name=assign_cells
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools
module load miniforge3
conda init bash
source ~/.bashrc
conda activate scanpy_env
echo "Active Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

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

profile_params_path="None"
bam_dir="${BASEDIR}/data/normal_cell_bams"
input_dir="${PROFILE_DIR}"
output_dir="${PROFILE_DIR}"
NCORES=32

log_dir="${PROFILE_DIR}/logs"
mkdir -p "$log_dir"
log_path="${log_dir}/${profile_name}_assign_cells_$(date +'%Y%m%d_%H%M%S').log"

python ${BASEDIR}/python_scripts/assign_cnv_profile_cells.py \
    --profile_name $profile_name \
    --profile_params_path $profile_params_path \
    --bam_dir $bam_dir \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --verbose \
    --ncores $NCORES \
    --allow_dup "True" \
    --group_name "all" \
    > "$log_path" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: Python script failed. Check log file for details."
    exit 1
fi
echo "Python script completed successfully."

echo "Current date and time: $(date)"
echo "Job completed successfully."