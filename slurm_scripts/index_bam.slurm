#!/bin/bash
#SBATCH --job-name=index_bams
#SBATCH --output=logs/output_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=sunge@mskcc.org

# Directory containing BAM files
BAM_DIR="/data1/shahs3/users/sunge/cnv_simulator/data/sample_3000cell_bams"

module load samtools

# Loop over all BAM files in the directory
for bam_file in "$BAM_DIR"/*.bam; do
    # Define expected index file
    index_file="${bam_file}.bai"

    # Check if the index file exists
    if [ ! -f "$index_file" ]; then
        echo "Indexing $bam_file..."
        samtools index "$bam_file"
    else
        echo "Index already exists for $bam_file. Skipping."
    fi
done

echo "Done indexing missing BAM files."