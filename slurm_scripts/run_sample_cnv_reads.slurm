#!/bin/bash
#SBATCH --job-name=sample_cnv_reads
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load samtools
module load java
module load miniforge3
conda init bash
source ~/.bashrc
conda activate scanpy_env
echo "Active Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Print current date and time
echo "Current date and time: $(date)"

PROFILE_NAME=$1
if [ -z "$PROFILE_NAME" ]; then
    echo "Error: No profile name provided."
    exit 1
fi
echo "Profile name: $PROFILE_NAME"

BASEDIR="/data1/shahs3/users/sunge/cnv_simulator"
PROFILE_DIR="${BASEDIR}/synthetic_bams_2/${PROFILE_NAME}"

log_dir="${PROFILE_DIR}/logs"
mkdir -p "$log_dir"
log_path="${log_dir}/${PROFILE_NAME}_sample_cnv_reads_$(date +'%Y%m%d_%H%M%S').log"

python /data1/shahs3/users/sunge/cnv_simulator/python_scripts/sample_cnv_reads.py \
    --profile_name $PROFILE_NAME \
    --profile_dir $PROFILE_DIR \
    --ncores $SLURM_CPUS_PER_TASK \
    > "$log_path" 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Python script failed. Check log file for details."
    exit 1
fi
echo "Python script completed successfully."

# Print current date and time
echo "Current date and time: $(date)"