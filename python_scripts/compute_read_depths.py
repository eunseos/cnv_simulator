#%%
import os
import sys
import logging

env_name = os.environ.get("CONDA_DEFAULT_ENV")
print(f"Current conda environment: {env_name}")
print(f"Python executable: {sys.executable}")

import pandas as pd
import numpy as np
import pysam
import argparse
from collections import defaultdict
from tqdm import tqdm
import scipy.sparse as sp


BASEDIR = "/data1/shahs3/users/sunge/cnv_simulator"

#%%

def get_cell_barcodes(cell_profile_path):
    """
    Get the cell barcodes from the cell profile file.
    """
    cell_profile_df = pd.read_csv(cell_profile_path, sep="\t")
    cell_barcodes_str = cell_profile_df[cell_profile_df["clone"] == -1]["cell_barcode"].iloc[0]
    cell_barcodes = [x.strip() for x in cell_barcodes_str.split(",")]
    return cell_barcodes

def parse_bins(gene_windows_path, chr_set):
    """
    Parse the gene windows file to get the bins.
    """
    bins = []
    with open(gene_windows_path) as f:
        for line in f:
            chrom, start, end = line.strip().split()[:3]
            if chrom in chr_set:
                bins.append((chrom, int(start), int(end)))
    return bins

def parse_gc_wig_bins(gc_wig_path, chr_set=None):
    """
    Parse the GC wig file to get bins as (chrom, start, end) tuples.
    If chr_set is provided, only include bins for those chromosomes.
    Bins are 1-based, inclusive: (start, end) = (1, 500000), (500001, 1000000), ...
    """
    bins = []
    with open(gc_wig_path) as f:
        for line in f:
            if line.startswith("fixedStep"):
                tokens = line.strip().split()
                chrom = tokens[1].split("=")[1]
                start = int(tokens[2].split("=")[1])  # keep as 1-based
                step = int(tokens[3].split("=")[1])
                span = int(tokens[4].split("=")[1]) if len(tokens) > 4 else step
                if chr_set is not None and chrom not in chr_set:
                    skip = True
                else:
                    skip = False
                curr_start = start
            elif not line.strip() or skip:
                continue
            else:
                # Each line is a value for the current bin
                bins.append((chrom, curr_start, curr_start + span - 1))
                curr_start += step
    return bins

def parse_gene_windows_bins(gene_windows_path, chr_set=None):
    """
    Parse the gene windows file to get bins as (chrom, start, end) tuples.
    If chr_set is provided, only include bins for those chromosomes.
    """
    bins = []
    with open(gene_windows_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            chrom, start, end = line.strip().split()[:3]
            if chr_set is None or chrom in chr_set:
                bins.append((chrom, int(start), int(end)))
    return bins


def compute_read_depths(profile_dir, profile_name, bam_filename, bin_size, bins):
    bam_file = f"{profile_dir}/{bam_filename}"
    cell_profile_path = f"{profile_dir}/{profile_name}_cell_profile.tsv"

    logger = logging.getLogger(__name__)
    logger.info(f"Computing read depths for {bam_file}...")

    # Get list of chromosomes from the BAM file
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        chr_set = set(bam.references)
        logger.info(f"Chromosomes in BAM file: {chr_set}")

    cell_barcodes = get_cell_barcodes(cell_profile_path)
    cell_idx_map = {cb: i for i, cb in enumerate(cell_barcodes)}
    logger.info(f"Number of cells: {len(cell_barcodes)}")
    logger.info(f"Number of bins: {len(bins)}")

    # Create a sparse matrix to store the read depths
    num_cells = len(cell_barcodes)
    num_bins = len(bins)

    data = []
    row_idx = []
    col_idx = []

    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for bin_i, (chrom, start, end) in tqdm(enumerate(bins), total = num_bins, desc = "Processing bins"):
            counts = defaultdict(int)
            for read in bam.fetch(chrom, start, end):
                if read.is_secondary or read.is_supplementary:
                    continue
                if read.has_tag("CB"):
                    cb = read.get_tag("CB")
                    if cb in cell_idx_map:
                        counts[cb] += 1
            # Ensure every cell has a value (zero if missing)
            for cb in cell_barcodes:
                row_idx.append(bin_i)
                col_idx.append(cell_idx_map[cb])
                data.append(counts.get(cb, 0))

    coverage_matrix = sp.coo_matrix((data, (row_idx, col_idx)), shape=(num_bins, num_cells))

    return coverage_matrix, bins, cell_barcodes


def get_hmmcopy_reads(coverage_matrix, bins, cell_barcodes, bin_size=5000, aggr_bin_size=500000):
    # Create bins DataFrame
    bins_df = pd.DataFrame(bins, columns=["chr", "start", "end"])

    nrows, ncols = coverage_matrix.shape
    bin_mult = aggr_bin_size // bin_size

    usable_rows = nrows - (nrows % bin_mult)

    if sp.issparse(coverage_matrix):
        coverage_matrix = coverage_matrix.tocsr()[:usable_rows, :]
        binned_coverage = coverage_matrix.toarray().reshape(-1, bin_mult, ncols).sum(axis=1)
    else:
        binned_coverage = coverage_matrix[:usable_rows, :].reshape(-1, bin_mult, ncols).sum(axis=1)

    # Aggregate bin metadata
    binned_bins = bins_df.iloc[:usable_rows, :].groupby(bins_df.index[:usable_rows] // bin_mult).agg({
        "chr": "first",
        "start": "first",
        "end": "last"
    }).reset_index(drop=True)

    # Flatten into long-form DataFrame
    long_data = []
    for bin_idx, bin_row in binned_bins.iterrows():
        for cell_idx, cell_id in enumerate(cell_barcodes):
            reads = binned_coverage[bin_idx, cell_idx]
            long_data.append({
                "cell_id": cell_id,
                "chr": bin_row["chr"],
                "start": bin_row["start"],
                "end": bin_row["end"],
                "reads": reads
            })

    long_df = pd.DataFrame(long_data)
    return long_df


#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Compute read depths from BAM files.")
    parser.add_argument("--profile_dir", type=str, required=True, help="Directory containing all profile files.")
    parser.add_argument("--profile_name", type=str, required=True, help="Name of the profile file.")
    parser.add_argument("--bam_filename", type=str, required=True, help="Name of the BAM file.")
    parser.add_argument("--gene_windows", type=str, default=None, help="Path to gene windows file for binning.")
    parser.add_argument("--gc_wig", type=str, default=None, help="Path to GC wig file for binning.")
    parser.add_argument("--bin_size", type=int, default=5000, help="Size of the bins in base pairs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    return parser.parse_args()

def main():
    args = parse_args()
    profile_dir = args.profile_dir
    profile_name = args.profile_name
    bam_filename = args.bam_filename
    gene_windows_path = args.gene_windows
    gc_wig_path = args.gc_wig
    bin_size = args.bin_size
    verbose = args.verbose

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Check if the profile directory exists
    if not os.path.exists(profile_dir):
        logging.error(f"Profile directory {profile_dir} does not exist.")
        sys.exit(1)

    logger.info(f"Using profile directory: {profile_dir}")
    logger.info(f"Using profile name: {profile_name}")

    # Get chromosomes from BAM
    with pysam.AlignmentFile(f"{profile_dir}/{bam_filename}", "rb") as bam:
        chr_set = set(bam.references)
    if gc_wig_path is None and gene_windows_path is None:
        logger.error("Either --gene_windows or --gc_wig must be provided.")
        sys.exit(1)
    if gene_windows_path is not None:
        bins = parse_gene_windows_bins(gene_windows_path, chr_set)
    elif gc_wig_path is not None:
        bins = parse_gc_wig_bins(gc_wig_path, chr_set)

    bam_path = f"{profile_dir}/{bam_filename}"
    if not os.path.exists(bam_path):
        logger.error(f"BAM file {bam_path} does not exist.")
        sys.exit(1)
    # Check if the BAM file is indexed
    if not os.path.exists(f"{bam_path}.bai"):
        logger.info(f"Indexing BAM file {bam_path}...")
        try:
            pysam.index(bam_path)
        except Exception as e:
            logger.error(f"Failed to index BAM file {bam_path}: {e}")
            sys.exit(1)

    depth_output_path = f"{profile_dir}/{bam_filename}.{bin_size}_read_depth.npz"
    bins_output_path = f"{profile_dir}/{bam_filename}.{bin_size}_bins.tsv"
    barcodes_output_path = f"{profile_dir}/{bam_filename}.{bin_size}_cell_barcodes.tsv"

    if os.path.exists(depth_output_path) and os.path.exists(bins_output_path) and os.path.exists(barcodes_output_path):
        logger.info(f"Read depth matrix already exists at {depth_output_path}. Skipping computation.")

        coverage_matrix = sp.load_npz(depth_output_path)
        with open(bins_output_path, "r") as f:
            bins = [(chrom, int(start), int(end)) for chrom, start, end in (line.strip().split("\t") for line in f)]
        with open(barcodes_output_path, "r") as f:
            cell_barcodes = [line.strip() for line in f.readlines()]

    else:
        logger.info(f"Computing read depths for {bam_filename} with bin size {bin_size}...")
        coverage_matrix, bins, cell_barcodes = compute_read_depths(profile_dir, profile_name, bam_filename, bin_size, bins)

        sp.save_npz(depth_output_path, coverage_matrix)
        logger.info(f"Saved read depth matrix to {depth_output_path}")

        # Save the cell barcodes and bin information
        with open(bins_output_path, "w") as f:
            for chrom, start, end in bins:
                f.write(f"{chrom}\t{start}\t{end}\n")
        logger.info(f"Saved bin information to {bins_output_path}")

        with open(barcodes_output_path, "w") as f:
            for cb in cell_barcodes:
                f.write(f"{cb}\n")
        logger.info(f"Saved cell barcodes to {barcodes_output_path}")

    hmmcopy_reads_path = f"{profile_dir}/{bam_filename}.{bin_size}_hmmcopy_reads.csv.gz"
    if os.path.exists(hmmcopy_reads_path):
        logger.info(f"HMMcopy reads already exist at {hmmcopy_reads_path}. Skipping computation.")
        return
    else:
        logger.info("Computing HMMcopy reads...")
        long_df = get_hmmcopy_reads(coverage_matrix, bins, cell_barcodes, bin_size=bin_size, aggr_bin_size=500000)
        long_df.to_csv(hmmcopy_reads_path, index=False, compression = "gzip")


if __name__ == "__main__":
    main()