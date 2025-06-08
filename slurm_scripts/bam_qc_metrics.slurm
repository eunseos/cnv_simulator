#!/bin/bash
#SBATCH --job-name=picard_metrics
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=componc_cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sunge@mskcc.org

module load java
module load miniforge3
conda init bash
source ~/.bashrc
conda activate R_env

SAMPLE_NAME=$1
PICARD_JAR="/home/sunge/packages/picard.jar"
BAMDIR="/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/${SAMPLE_NAME}"
REF_FASTA="/data1/shahs3/users/sunge/cnv_simulator/data/refs_2/Homo_sapiens.GRCh37.dna.primary_assembly_reordered.fa"

OUTDIR="${BAMDIR}/qc_metrics"
if [ ! -d "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
fi

BAM_FILE="${BAMDIR}/${SAMPLE_NAME}_final_sorted_cnv.bam"
BASELINE_FILE="${BAMDIR}/${SAMPLE_NAME}_baseline_cells.bam"
if [ ! -f "$BAM_FILE" ]; then
    echo "Error: BAM file not found: $BAM_FILE"
    exit 1
fi

# Collect WGS metrics (depth + uniformity)
WGS_OUT="${OUTDIR}/${SAMPLE_NAME}_wgs_metrics.txt"
WGS_HIST_OUT="${OUTDIR}/${SAMPLE_NAME}_wgs_histogram.pdf"
java -Xmx32g -jar $PICARD_JAR CollectWgsMetrics \
    I=$BAM_FILE \
    O=$WGS_OUT\
    R=$REF_FASTA \
    INCLUDE_BQ_HISTOGRAM=true

WGS_BASE_OUT="${OUTDIR}/${SAMPLE_NAME}_baseline_wgs_metrics.txt"
WGS_BASE_HIST_OUT="${OUTDIR}/${SAMPLE_NAME}_baseline_wgs_histogram.pdf"
java -Xmx32g -jar $PICARD_JAR CollectWgsMetrics \
    I=$BASELINE_FILE \
    O=$WGS_BASE_OUT\
    R=$REF_FASTA \
    INCLUDE_BQ_HISTOGRAM=true

# GC bias metrics
GC_OUT="${OUTDIR}/${SAMPLE_NAME}_gc_bias_metrics.txt"
GC_CHART_OUT="${OUTDIR}/${SAMPLE_NAME}_gc_bias_chart.pdf"
GC_SUMMARY_OUT="${OUTDIR}/${SAMPLE_NAME}_gc_bias_summary.txt"
java -Xmx32g -jar $PICARD_JAR CollectGcBiasMetrics \
    I=$BAM_FILE \
    O=$GC_OUT \
    R=$REF_FASTA \
    CHART=$GC_CHART_OUT \
    S=$GC_SUMMARY_OUT

GC_BASE_OUT="${OUTDIR}/${SAMPLE_NAME}_baseline_gc_bias_metrics.txt"
GC_BASE_CHART_OUT="${OUTDIR}/${SAMPLE_NAME}_baseline_gc_bias_chart.pdf"
GC_BASE_SUMMARY_OUT="${OUTDIR}/${SAMPLE_NAME}_baseline_gc_bias_summary.txt"
java -Xmx32g -jar $PICARD_JAR CollectGcBiasMetrics \
    I=$BASELINE_FILE \
    O=$GC_BASE_OUT \
    R=$REF_FASTA \
    CHART=$GC_BASE_CHART_OUT \
    S=$GC_BASE_SUMMARY_OUT