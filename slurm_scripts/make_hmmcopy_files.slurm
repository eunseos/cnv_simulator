#!/bin/bash
#SBATCH --job-name=bam_hmmcopy
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load miniforge3
conda init bash
source ~/.bashrc
conda activate seqtools_env

HMMCOPY_DIR="/home/sunge/packages/hmmcopy_utils"
REF_DIR="/data1/shahs3/users/sunge/cnv_simulator/data/refs"
BAM_DIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2"

fasta_path="${REF_DIR}/Homo_sapiens.GRCh37.dna.primary_assembly.fa"
bw_path="${REF_DIR}/Homo_sapiens.GRCh37.fa.map.bw"
map_wig_path="${REF_DIR}/Homo_sapiens.GRCh37.map.ws_500000.wig"
gc_path="${REF_DIR}/Homo_sapiens.GRCh37.gc_500000.wig"

# Make bowtie index (index_hmmcopy)
# $HMMCOPY_DIR/util/mappability/generateMap.pl -b $fasta_path -o $bw_path
# echo "Bowtie index created"

# Make map (map_hmmcopy)
# $HMMCOPY_DIR/util/mappability/generateMap.pl -c 1,2,5,7,12 $fasta_path -o $bw_path -w 125
# echo "Map created at ${bw_path}"

# Convert to wig file with mapCounter (wig_hmmcopy)
# $HMMCOPY_DIR/bin/mapCounter -w 500000 -c 1,2,5,7,12 $bw_path > $map_wig_path
# echo "Mapcounter complete at ${map_wig_path}"

# Make GC content file (gc_hmmcopy)
# $HMMCOPY_DIR/bin/gcCounter -w 500000 -c 1,2,5,7,12 $fasta_path > $gc_path
# echo "GC wig created at ${gc_path}"

# Count reads in BAM files (bam_hmmcopy)
bam_path="${BAM_DIR}/$1/$1_final_sorted_cnv.bam"
if [ ! -f "$bam_path" ]; then
    echo "Error: BAM file not found: $bam_path"
    exit 1
fi
readcounts_path="${BAM_DIR}/$1/$1_500000.readcounts.seg"
$HMMCOPY_DIR/bin/readCounter -w 500000 $bam_path > $readcounts_path
echo "BAM readcounts complete at ${readcounts_path}"

baseline_bam_path="${BAM_DIR}/$1/$1_baseline_cells.bam"
if [ ! -f "$baseline_bam_path" ]; then
    echo "Error: Baseline BAM file not found: $baseline_bam_path"
    exit 1
fi
baseline_readcounts_path="${BAM_DIR}/$1/$1_baseline_500000.readcounts.seg"
$HMMCOPY_DIR/bin/readCounter -w 500000 $baseline_bam_path > $baseline_readcounts_path
echo "Baseline BAM readcounts complete at ${baseline_readcounts_path}"