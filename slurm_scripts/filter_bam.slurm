#!/bin/bash
#SBATCH --job-name=filter_bam_sample3
#SBATCH --output=logs/output_%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=sunge@mskcc.org

module load samtools

DATADIR="/data1/shahs3/users/sunge/cnv_simulator/data"
barcode_dir="${DATADIR}/sample_normal_cells"

bam_save_dir="${DATADIR}/sample_3000cell_bams"
mkdir -p "$bam_save_dir"

barcode_file="${barcode_dir}/SPECTRUM-OV-110_S1_CD45N_BOWEL_130424A_L2_cell_barcodes.txt"
bam_file="/data1/shahs3/isabl_data_lake/analyses/90/85/29085/results/SHAH_H000706_T03_05_DLP01_all_cells_bulk.bam"
sample_name=$(basename "$bam_file" .bam)
echo "Processing sample: $sample_name with barcode file: $barcode_file"

# Build samtools view filter
barcode_filters=$(awk '{print "-d CB:"$1}' "$barcode_file" | tr '\n' ' ')

# Filter and output
samtools view -@ 32 -b $barcode_filters "$bam_file" -o "${bam_save_dir}/${sample_name}_normal_cells.bam"
