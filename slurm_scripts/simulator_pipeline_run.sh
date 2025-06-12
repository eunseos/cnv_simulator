#!/bin/bash
#SBATCH --job-name=cnv_simulator_pipeline
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=1-00:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

test_name=$1
echo "Running CNV simulator pipeline for test: $test_name"

# # Assign cells to cnv profiles
assign_cells_jid=$(sbatch run_assign_cnv_profile_cells.slurm $test_name | awk '{print $4}')
echo "Submitted assign_cells with job ID $assign_cells_jid"

# # Generate BAM files for each profile
generate_bams_jid=$(sbatch --dependency=afterok:$assign_cells_jid make_baseline_bam.slurm $test_name | awk '{print $4}')
# generate_bams_jid=$(sbatch make_baseline_bam.slurm $test_name | awk '{print $4}')
echo "Submitted generate_bams with job ID $generate_bams_jid"

# Sample reads for each profile
sample_reads_jid=$(sbatch --dependency=afterok:$generate_bams_jid run_sample_cnv_reads.slurm $test_name | awk '{print $4}')
# sample_reads_jid=$(sbatch run_sample_cnv_reads.slurm $test_name | awk '{print $4}')
echo "Submitted sample_reads with job ID $sample_reads_jid"

# Get read depth matrices for each profile
read_depths_jid=$(sbatch --dependency=afterok:$sample_reads_jid run_compute_read_depths.slurm $test_name | awk '{print $4}')
echo "Submitted read_depths with job ID $read_depths_jid"

# Print done message
echo "All jobs submitted successfully. Pipeline for test '$test_name' is running."