#%%
import os
import sys
import logging

env_name = os.environ.get("CONDA_DEFAULT_ENV")
print(f"Current conda environment: {env_name}")

print(sys.executable)

import pandas as pd
import numpy as np
import pysam
import argparse
import subprocess
import tempfile
import shutil

BASEDIR = "/data1/shahs3/users/sunge/cnv_simulator"
OUTDIR = f"{BASEDIR}/synthetic_bams"

VERBOSE = True
NCORES = 32

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#%%
########################################################################
### Samtools Helper Functions
########################################################################

def samtools_read_id_filter(bam_path, out_path, read_ids):
    """
    Saves new bam file with only the specified read IDs.
    """
    with tempfile.NamedTemporaryFile(mode = "w", delete = False) as tmp:
        for rid in read_ids:
            tmp.write(f"{rid}\n")
        tmp_path = tmp.name
        if VERBOSE:
            logger.debug(f"Temporary file created: {tmp_path}")

    # Print number of read IDs to be filtered
    if VERBOSE:
        logger.debug(f"Number of read IDs to filter: {len(read_ids)}")
        logger.debug(f"Read IDs: {read_ids[:10]}...")

    try:
        cmd = [
            "samtools", "view", "-@", str(NCORES),
            "-b", "-N", tmp_path, bam_path,
            "-o", out_path
        ]
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check = True)
    finally:
        os.remove(tmp_path)
        if VERBOSE:
            logger.debug(f"Temporary file deleted: {tmp_path}")
    
    return out_path

def samtools_get_cell_reads(bam_path, chr = None, start = None, end = None):
    """
    Get the reads in a specific region sorted by cell barcode.

    Parameters:
    ----------
    bam_path (str): Path to source BAM file.
    chr (str): Chromosome name
    start (int): Start position of the region.
    end (int): End position of the region.

    Returns:
    -------
    cell_reads_dict: Dictionary with cell barcodes as keys and lists of read IDs as values.
    """
    cmd = [
        "samtools", "view", "-@", str(NCORES),
        bam_path
    ]
    if chr and start and end:
        cmd.append(f"{chr}:{start}-{end}")
    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        region_reads_result = subprocess.run(cmd, stdout=subprocess.PIPE, check = True, text = True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e.stderr}")
        raise

    cell_reads_dict = {}
    reads_cell_dict = {}
    for line in region_reads_result.stdout.splitlines():
        fields = line.strip().split("\t")
        read_id = fields[0]
        cb_tag = next((f for f in fields[11:] if f.startswith("CB:Z:")), None)
        if cb_tag:
            cell_barcode = cb_tag.split(":")[2]
            cell_reads_dict.setdefault(cell_barcode, []).append(read_id)
            reads_cell_dict[read_id] = cell_barcode
    return cell_reads_dict, reads_cell_dict


def samtools_get_group_read_count(group_bam_dict, chr, start, end):
    """
    Get the number of reads in the group_bam_dict for given region.
    """
    group_bam_length_dict = {}
    for group, bam_path in group_bam_dict.items():
        group_bam_length_dict[group] = samtools_get_indexed_read_count(bam_path, chr, start, end)
    return group_bam_length_dict


def samtools_get_indexed_read_count(bam_path, chr, start, end, cell_barcodes = None):
    """
    Get the number of reads in a specific region.

    Parameters:
    ----------
    bam_path (str): Path to source BAM file.
    chr (str): Chromosome name
    start (int): Start position of the region.
    end (int): End position of the region.

    Returns:
    -------
    read_count: Number of reads in the specified region.
    """
    cmd = [
        "samtools", "view", "-@", str(NCORES),
        "-c"
    ]

    if cell_barcodes:
        for cb in cell_barcodes:
            cmd.extend(["-d", f"CB:{cb}"])

    cmd.append(bam_path)
    cmd.append(f"{chr}:{start}-{end}")

    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        region_reads_result = subprocess.run(cmd, stdout=subprocess.PIPE, check = True, text = True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise

    read_count = int(region_reads_result.stdout.strip())
    return read_count


def samtools_get_unindexed_read_count(bam_path):
    """
    Get the number of reads in a specific region.

    Parameters:
    ----------
    bam_path (str): Path to source BAM file.
    chr (str): Chromosome name
    start (int): Start position of the region.
    end (int): End position of the region.

    Returns:
    -------
    read_count: Number of reads in the specified region.
    """
    cmd = f"samtools view {bam_path} | wc -l"
    try:
        if VERBOSE:
            logger.debug(f"Executing command: {cmd}")
        region_reads_result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, check = True, text = True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {cmd}")
        logger.error(f"Error message: {e}")
        raise
    read_count = int(region_reads_result.stdout.strip())
    return read_count


def samtools_get_entire_read_count(bam_path):
    """
    Returns the total number of mapped reads in a BAM file using `samtools idxstats`.

    Parameters:
    ----------
    bam_path (str): Path to the BAM file (must be indexed with .bai)

    Returns:
    -------
    int: Total number of mapped reads
    """
    cmd = ["samtools", "idxstats", bam_path]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        total_mapped_reads = 0
        for line in result.stdout.strip().split("\n"):
            fields = line.split("\t")
            if len(fields) >= 3:
                mapped = int(fields[2])
                total_mapped_reads += mapped
        return total_mapped_reads
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running samtools idxstats: {e.stderr}")
        raise


def samtools_sample_reads(bam_path, out_path, frac_reads, region = None, cell_barcodes = None,
                          seed=5091130):
    """
    Sample a fraction of reads from a specific region of a BAM file.
    
    Parameters:
    - bam_path: Input BAM file
    - out_path: Output BAM file
    - frac_reads: Fraction of reads to sample (0.0 to 1.0)
    - region: Genomic region string, e.g., 'chr1:10000-50000'. If None, samples from the whole file.
    - cell_barcodes: List of cell barcodes to sample from. If None, samples from all reads.
    - seed: Random seed for reproducibility
    """

    cmd = [
        "samtools", "view", "-@", str(NCORES),
        "--subsample-seed", str(seed),
        "--subsample", str(frac_reads),
        "-b"
    ]

    if cell_barcodes:
        for cb in cell_barcodes:
            cmd.extend(["-d", f"CB:{cb}"])

    cmd.append(bam_path)

    if region:
        cmd.append(region)

    cmd.extend(["-o", out_path])

    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise
    return out_path


def samtools_merge(bam_paths, out_path):
    """
    Merge multiple BAM files into a single BAM file.

    Parameters:
    ----------
    bam_paths (list): List of paths to the BAM files to merge.
    out_path (str): Path to the output merged BAM file.

    Returns:
    -------
    out_path: Path to the merged BAM file.
    """
    cmd = [
        "samtools", "merge", "-@", str(NCORES),
        out_path,
        *bam_paths
    ]
    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e.stderr.strip()}")
        raise
    return out_path

def samtools_index(bam_path):
    """
    Index a BAM file.

    Parameters:
    ----------
    bam_path (str): Path to the BAM file to index.

    Returns:
    -------
    None
    """
    cmd = [
        "samtools", "index", "-@", str(NCORES),
        bam_path
    ]
    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise

def samtools_sort(bam_path, out_path):
    """
    Sort a BAM file.

    Parameters:
    ----------
    bam_path (str): Path to the BAM file to sort.
    out_path (str): Path to the output sorted BAM file.

    Returns:
    -------
    None
    """
    cmd = [
        "samtools", "sort", "-@", str(NCORES),
        "-o", out_path,
        bam_path
    ]
    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise
    return out_path

##########################################################################
### General Functions
##########################################################################

def get_group_bam_dict(profile_df):
    """
    Get a dictionary of group names and their corresponding bam file paths.
    """
    group_bam_dict = {}
    unique_groups = profile_df["cell_group"].str.split(",").explode().unique()

    for group in unique_groups:
        # Get the bam file path for the group
        if not group.isdigit():
            bam_path = f"{BASEDIR}/data/normal_cell_bams/{group}.bam"
        else:
            bam_path = f"{BASEDIR}/data/normal_cell_bams/group_{group}_merged.bam"
        if not os.path.exists(bam_path):
            raise ValueError(f"BAM file does not exist: {bam_path}")
        group_bam_dict[group] = bam_path

    return group_bam_dict


def assign_new_barcodes(bam_path, baseline_clone_cell_barcodes, output_bam_path):
    
    bamfile = pysam.AlignmentFile(bam_path, "rb")
    out_bam = pysam.AlignmentFile(output_bam_path, "wb", template=bamfile)

    new_cell_assignments = {}
    for i, read in enumerate(bamfile):
        read_cell = read.get_tag("CB")
        if read_cell not in new_cell_assignments:
            new_cb = np.random.choice(baseline_clone_cell_barcodes)
            new_cell_assignments[read_cell] = new_cb
        else:
            new_cb = new_cell_assignments[read_cell]
        read.set_tag("CB", new_cell_assignments[read_cell])
        out_bam.write(read)
    bamfile.close()
    out_bam.close()
    if VERBOSE:
        logger.debug(f"Temporary BAM file with renamed cell barcodes created: {output_bam_path}")
    return


def get_sample_read_counts(profile_row, new_group_cell_dict, baseline_cell_reads_dict, clone_cell_barcodes, group_bam_dict):
    """
    Get the number of reads to sample from the baseline bam file and the group bam files for a clone in a region.
    
    Parameters:
    profile_row: Row in the profile dataframe for the region
    new_group_cell_dict: Dictionary of groups and their corresponding cell barcodes (new cells)
    baseline_cell_reads_dict: Dictionary of cell barcodes and their corresponding read IDs in the baseline bam file
    clone_cell_barcodes: List of cell barcodes for the clone
    group_bam_dict: Dictionary of groups and their corresponding bam file paths

    Returns:
    n_clone_baseline_reads: # reads for clone in baseline bam file for this region
    n_clone_additional_reads: # reads to sample from group bam file for clone in this region
    n_clone_additional_reads_per_group: # reads to sample from each group bam file for clone in this region
    n_clone_frac_reads_per_group: # fraction of reads to sample from each group bam file for clone in this region
    """

    # Compute proportion of additional cells in each group
    group_cell_proportions = np.array([len(new_group_cell_dict[group]) for group in new_group_cell_dict], dtype = float)
    group_cell_proportions /= group_cell_proportions.sum()
    if VERBOSE:
        logger.debug(f"Group proportions: {group_cell_proportions}")

    # Figure out how many additional reads to sample based on number of reads in baseline bam file in region for clone
    n_clone_baseline_reads = sum(len(reads) for cb, reads in baseline_cell_reads_dict.items() if cb in clone_cell_barcodes)
    n_clone_additional_reads = (n_clone_baseline_reads // 2) * profile_row.copy_number - n_clone_baseline_reads

    # Calculate the number of additional reads to sample from each group based on proportion of cells from each group
    n_clone_additional_reads_per_group_array = np.random.multinomial(n_clone_additional_reads, group_cell_proportions)
    n_clone_additional_reads_per_group = dict(zip(new_group_cell_dict.keys(), n_clone_additional_reads_per_group_array))
    # Check that the total number of additional reads is equal to the number of reads to sample
    if sum(n_clone_additional_reads_per_group.values()) != n_clone_additional_reads:
        raise ValueError(f"Total number of additional reads does not match: {sum(n_clone_additional_reads_per_group.values())} != {n_clone_additional_reads}")
    
    # Need required reads per group / total reads per group
    frac_reads_per_group = {}
    for group, n_reads in n_clone_additional_reads_per_group.items():
        # Total reads in group bam file for this region, new additional cells
        total_reads_in_group = samtools_get_indexed_read_count(group_bam_dict[group], profile_row.chr,
                                                               profile_row.start, profile_row.end,
                                                               cell_barcodes = new_group_cell_dict[group])
        if VERBOSE:
            logger.debug(f"Total reads in group {group}: {total_reads_in_group}")
        if total_reads_in_group == 0:
            raise ValueError(f"Total reads in group {group} is 0: {total_reads_in_group}")
        frac_reads_per_group[group] = n_reads / total_reads_in_group
        if frac_reads_per_group[group] > 1:
            raise ValueError(f"Fraction of reads to sample from group {group} is greater than 1: {frac_reads_per_group[group]}")
        if VERBOSE:
            logger.debug(f"Fraction of reads to sample from group {group}: {frac_reads_per_group[group]}")

    return n_clone_baseline_reads, n_clone_additional_reads, n_clone_additional_reads_per_group, frac_reads_per_group


def remove_reads_from_baseline_bam(baseline_bam_path, profile_row, profile_row_index, clone_row, 
                                   TMP_BAMDIR):
    
    clone_cell_barcodes = clone_row.cell_barcode.split(",")
    
    out_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_tmp.bam"
    
    if profile_row.copy_number == 0:
        # No bam file for this region
        samtools_sample_reads(baseline_bam_path, out_bam_path, 0.0,
                              region = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}",
                              cell_barcodes = clone_cell_barcodes)
    elif profile_row.copy_number == 1:
        # Remove half reads from this region in the baseline bam file
        samtools_sample_reads(baseline_bam_path, out_bam_path, 0.5,
                              region = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}",
                              cell_barcodes = clone_cell_barcodes)
    elif profile_row.copy_number == 2:
        # Select all reads from this region in the baseline bam file
        samtools_sample_reads(baseline_bam_path, out_bam_path, 1.0,
                              region = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}",
                              cell_barcodes = clone_cell_barcodes)
    else:
        ValueError(f"Invalid copy number: {profile_row.copy_number}")

    if os.path.exists(out_bam_path):
        n_reads = samtools_get_unindexed_read_count(out_bam_path)
        if VERBOSE:
            logger.debug(f"Number of reads in baseline bam file: {n_reads}")
    else:
        n_reads = 0

    return out_bam_path, n_reads
    

def add_reads_to_baseline_bam(baseline_bam_path, group_bam_dict, 
                              profile_row, profile_row_index, clone_row, TMP_BAMDIR):
    # Check that copy number is greater than 0
    if profile_row.copy_number <= 0:
        raise ValueError(f"Copy number must be greater than 0: {profile_row.copy_number}")

    # Get the cell barcodes for the clone
    clone_cell_barcodes = clone_row.cell_barcode.split(",")
    baseline_cell_reads_dict, _ = samtools_get_cell_reads(baseline_bam_path, profile_row.chr,
                                                          profile_row.start, profile_row.end)
    missing_clone_barcodes = [cb for cb in clone_cell_barcodes if cb not in baseline_cell_reads_dict]
    if missing_clone_barcodes:
        print(clone_cell_barcodes)
        print(baseline_cell_reads_dict)
        raise ValueError(f"Missing cell barcodes in baseline BAM: {missing_clone_barcodes}")

    # Get all necessary groups for this region, assign cell barcodes to groups
    new_groups = profile_row.cell_group.split(",")
    new_cell_barcodes = profile_row.cell_barcode.split(",")
    new_group_cell_dict = {} # group: [cell_barcode in row]
    for group, cell_barcode in zip(new_groups, new_cell_barcodes):
        if group not in group_bam_dict:
            raise ValueError(f"BAM file for group {group} does not exist.")
        new_group_cell_dict.setdefault(group, []).append(cell_barcode)

    # Get the number of reads in the baseline bam file for this region
    n_baseline_reads, n_additional_reads, n_additional_reads_per_group, frac_reads_per_group = get_sample_read_counts(
        profile_row, new_group_cell_dict, baseline_cell_reads_dict, clone_cell_barcodes, group_bam_dict)

    if VERBOSE:
        logger.debug(f"Region {profile_row.chr}:{profile_row.start}-{profile_row.end} has {n_baseline_reads} reads in baseline bam file")
        logger.debug(f"Sampling {n_additional_reads} additional reads")
        logger.debug(f"Number of additional reads per group: {n_additional_reads_per_group}")

    # For each group, get the bam file and sample reads from the group bam file
    tmp_group_bam_paths = [] # 1 temporary bam file per group
    for group, frac_reads in frac_reads_per_group.items():
        print(f"Sampling {frac_reads} fraction from group {group}")
        region_str = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}"

        # Sample reads from the group bam file
        sampled_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_{group}_tmp.bam"
        samtools_sample_reads(group_bam_dict[group], sampled_bam_path, frac_reads,
                              region = region_str, cell_barcodes = new_group_cell_dict[group])
        tmp_group_bam_paths.append(sampled_bam_path)

        if VERBOSE:
            logger.debug(f"Temporary BAM file created: {sampled_bam_path}")
    
    # Merge the temporary bam files into a single bam file
    merged_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_merged_tmp.bam"
    if len(tmp_group_bam_paths) < 2:
        n_tmp_reads = samtools_get_unindexed_read_count(tmp_group_bam_paths[0])
        if VERBOSE:
            logger.debug(f"# reads in merged bam: {n_tmp_reads}, # additional reads: {n_additional_reads}")
        shutil.move(tmp_group_bam_paths[0], merged_bam_path)
    else:
        samtools_merge(tmp_group_bam_paths, merged_bam_path)
        if VERBOSE:
            logger.debug(f"Merged BAM file created: {merged_bam_path}")

    # Check that the number of reads in the merged bam file is equal to the number of additional reads
    n_merged_reads = samtools_get_unindexed_read_count(merged_bam_path)
    if VERBOSE:
        logger.debug(f"Number of reads in merged bam file: {n_merged_reads}")

    # Remove the temporary bam files
    for tmp_bam_path in tmp_group_bam_paths:
        if os.path.exists(tmp_bam_path):
            os.remove(tmp_bam_path)
            if VERBOSE:
                logger.debug(f"Temporary BAM file deleted: {tmp_bam_path}")

    # Replace the cell barcodes in the merged bam file with the new cell barcodes
    renamed_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_renamed_tmp.bam"
    assign_new_barcodes(merged_bam_path, clone_cell_barcodes, renamed_bam_path)
    
    if VERBOSE:
        logger.debug(f"Temporary BAM file with renamed cell barcodes created: {renamed_bam_path}")

    # Remove the merged bam file
    os.remove(merged_bam_path)
    if VERBOSE:
        logger.debug(f"Merged BAM file deleted: {merged_bam_path}")
    
    # Get original reads from baseline bam file
    baseline_clone_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_baseline_tmp.bam"
    samtools_sample_reads(baseline_bam_path, baseline_clone_path, 1.0,
                          region = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}",
                          cell_barcodes = clone_cell_barcodes)
    
    out_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_tmp.bam"
    # Merge the renamed bam file with the baseline bam file
    samtools_merge([baseline_clone_path, renamed_bam_path], out_bam_path)
    if VERBOSE:
        logger.debug(f"Final BAM file created: {out_bam_path}")
    # Remove the renamed bam file
    os.remove(renamed_bam_path)
    if VERBOSE:
        logger.debug(f"Renamed BAM file deleted: {renamed_bam_path}")
    # Remove the baseline bam file
    os.remove(baseline_clone_path)
    if VERBOSE:
        logger.debug(f"Baseline BAM file deleted: {baseline_clone_path}")

    return out_bam_path, n_merged_reads


def normalize_final_bam(baseline_bam_path, bam_path, out_path, n_final_reads):

    n_baseline_reads = samtools_get_entire_read_count(baseline_bam_path)
    
    sample_frac = n_baseline_reads / n_final_reads
    if sample_frac > 1:
        logger.info(f"Sample fraction is greater than 1: {sample_frac}. Sampling from baseline bam file.")
        shutil.copy(baseline_bam_path, out_path)
    else:
        samtools_sample_reads(bam_path, out_path, sample_frac)

    if VERBOSE:
        logger.debug(f"Final BAM file created: {out_path}")

    return


def sample_cnv_reads(profile_name, group_name):

    # Get list of all groups, read in bam files, dictionary of group:bam path
    if group_name == "":
        prefix = f"{profile_name}"
    else:
        prefix = f"{profile_name}_{group_name}"
    profile_path = f"{BASEDIR}/data/small_cnv_profiles/{prefix}_cnv_profile.tsv"
    profile_df = pd.read_csv(profile_path, sep="\t")
    group_bam_dict = get_group_bam_dict(profile_df)

    # Get list of baseline cells, make bam file containing all reads from baseline cells
    baseline_bam_path = f"{OUTDIR}/{profile_name}_baseline_cells.bam"
    if not os.path.exists(baseline_bam_path):
        raise ValueError(f"Baseline BAM file does not exist: {baseline_bam_path}")
    
    TMP_BAMDIR = f"{OUTDIR}/{prefix}_intermediate_bams"
    os.makedirs(TMP_BAMDIR, exist_ok=True)

    cnv_profile_df = profile_df[(profile_df["clone"] != -1) & (profile_df["chr"] != 0)]
    
    # For every row in the profile df (except for first row, which is baseline cells)
    all_tmp_bam_paths = []
    total_reads = 0
    for index, row in cnv_profile_df.iterrows():
        # Get the clone row
        cur_row_clone = row["clone"]
        clone_row = profile_df.iloc[cur_row_clone + 1]
        cur_copy_number = row["copy_number"]

        logger.info(f"Processing row {index} for clone {cur_row_clone} with copy number {cur_copy_number}")

        if cur_copy_number <= 2:
            tmp_bam_path, readcount = remove_reads_from_baseline_bam(baseline_bam_path, row, index, clone_row, TMP_BAMDIR)
        else:
            tmp_bam_path, readcount = add_reads_to_baseline_bam(baseline_bam_path, group_bam_dict, row, index, clone_row, TMP_BAMDIR)
        all_tmp_bam_paths.append(tmp_bam_path)
        total_reads += readcount

    # Merge all the temporary bam files into a single bam file
    merged_bam_path = f"{OUTDIR}/{prefix}_unnormalized_cnv.bam"
    if len(all_tmp_bam_paths) < 2:
        n_tmp_reads = samtools_get_unindexed_read_count(all_tmp_bam_paths[0])
        if VERBOSE:
            logger.debug(f"# reads in merged bam: {n_tmp_reads}")
        shutil.move(all_tmp_bam_paths[0], merged_bam_path)
    else:
        samtools_merge(all_tmp_bam_paths, merged_bam_path)
        if VERBOSE:
            n_total_reads = samtools_get_unindexed_read_count(merged_bam_path)
            logger.debug(f"Total number of reads in merged bam file: {n_total_reads}")
            logger.debug(f"Sum of reads in temporary bam files: {total_reads}")
            logger.debug(f"Merged BAM file created: {merged_bam_path}")

    # Remove all files in intermediate bam directory
    if VERBOSE:
        logger.debug(f"Deleting temporary BAM files in {TMP_BAMDIR}")
    if os.path.exists(TMP_BAMDIR):
        shutil.rmtree(TMP_BAMDIR)
        if VERBOSE:
            logger.debug(f"Temporary BAM directory deleted: {TMP_BAMDIR}")

    # Sample original number of reads from the modified bam file
    if VERBOSE:
        logger.debug(f"Sampling {total_reads} reads from merged bam file")
    final_bam_path = f"{OUTDIR}/{prefix}_final_cnv.bam"
    normalize_final_bam(baseline_bam_path, merged_bam_path, final_bam_path, total_reads)

    # Sort the final bam file
    if VERBOSE:
        logger.debug(f"Sorting final BAM file: {final_bam_path}")
    sorted_bam_path = f"{OUTDIR}/{prefix}_final_sorted_cnv.bam"
    samtools_sort(final_bam_path, sorted_bam_path)

    # Index the final bam file
    if VERBOSE:
        logger.debug(f"Indexing final BAM file: {sorted_bam_path}")
    samtools_index(sorted_bam_path)

    print("DONE!")
    return


#%%

def parse_args():
    parser = argparse.ArgumentParser(description="Sample CNV reads from BAM files.")
    parser.add_argument("--profile_name", type=str, required=True, help="Name of the profile.")
    parser.add_argument("--group_name", type=str, default="", help="Name of the group.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    profile_name = args.profile_name
    group_name = args.group_name

    sample_cnv_reads(profile_name, group_name)