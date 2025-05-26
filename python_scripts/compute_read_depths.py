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

def compute_read_depths(profile_dir, profile_name, bam_filename, bin_size, gene_windows_path):
    bam_file = f"{profile_dir}/{bam_filename}"
    cell_profile_path = f"{profile_dir}/{profile_name}_cell_profile.tsv"
    depth_output_path = f"{profile_dir}/{bam_filename}.{bin_size}_read_depth.npz"
    bins_output_path = f"{profile_dir}/{bam_filename}.{bin_size}_bins.tsv"
    barcodes_output_path = f"{profile_dir}/{bam_filename}.{bin_size}_cell_barcodes.tsv"

    logger = logging.getLogger(__name__)
    logger.info(f"Computing read depths for {bam_file}...")

    # Get list of chromosomes from the BAM file
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        chr_set = set(bam.references)
        logger.info(f"Chromosomes in BAM file: {chr_set}")

    cell_barcodes = get_cell_barcodes(cell_profile_path)
    cell_idx_map = {cb: i for i, cb in enumerate(cell_barcodes)}
    bins = parse_bins(gene_windows_path, chr_set)
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
            for cb, count in counts.items():
                row_idx.append(bin_i)
                col_idx.append(cell_idx_map[cb])
                data.append(count)
    
    coverage_matrix = sp.coo_matrix((data, (row_idx, col_idx)), shape=(num_bins, num_cells))
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
    
    return


#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Compute read depths from BAM files.")
    parser.add_argument("--profile_dir", type=str, required=True, help="Directory containing all profile files.")
    parser.add_argument("--profile_name", type=str, required=True, help="Name of the profile file.")
    parser.add_argument("--bam_filename", type=str, required=True, help="Name of the BAM file.")
    parser.add_argument("--bin_size", type=int, default=10000, help="Size of the bins in base pairs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    return parser.parse_args()

def main():
    args = parse_args()
    profile_dir = args.profile_dir
    profile_name = args.profile_name
    bam_filename = args.bam_filename
    bin_size = args.bin_size
    verbose = args.verbose

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Check if the profile directory exists
    if not os.path.exists(profile_dir):
        logging.error(f"Profile directory {profile_dir} does not exist.")
        sys.exit(1)
    logger = logging.getLogger(__name__)

    logger.info(f"Using profile directory: {profile_dir}")
    logger.info(f"Using profile name: {profile_name}")

    gene_windows_path = f"{BASEDIR}/data/genome_{bin_size // 1000}kb_bins.bed"

    bam_path = f"{profile_dir}/{bam_filename}"
    if not os.path.exists(bam_path):
        logger.error(f"BAM file {bam_path} does not exist.")
        sys.exit(1)
    # Check if the BAM file is indexed
    if not os.path.exists(f"{bam_path}.bai"):
        logger.info(f"Indexing BAM file {bam_path}...")
        pysam.index(bam_path)
    
    cell_profile_path = f"{profile_dir}/{profile_name}_cell_profile.tsv"
    output_path = f"{profile_dir}/{bam_filename}.{bin_size}_read_depth.npz"

    compute_read_depths(profile_dir, profile_name, bam_filename, bin_size, gene_windows_path)

    return

if __name__ == "__main__":
    main()