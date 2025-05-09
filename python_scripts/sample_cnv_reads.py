#%%
import os
import sys
env_name = os.environ.get("CONDA_DEFAULT_ENV")
print(f"Current conda environment: {env_name}")

print(sys.executable)

import pandas as pd
import numpy as np
import pysam
import argparse
import subprocess
import tempfile

BASEDIR = "/data1/shahs3/users/sunge/cnv_simulator"
OUTDIR = f"{BASEDIR}/synthetic_bams"

VERBOSE = True
NCORES = 32

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
            print(f"Temporary file created: {tmp_path}")

    # Print number of read IDs to be filtered
    if VERBOSE:
        print(f"Number of read IDs to filter: {len(read_ids)}")
        print(f"Read IDs: {read_ids[:10]}...")

    try:
        cmd = [
            "samtools", "view", "-@", str(NCORES),
            "-b", "-N", tmp_path, bam_path,
            "-o", out_path
        ]
        if VERBOSE:
            print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check = True)
    finally:
        os.remove(tmp_path)
        if VERBOSE:
            print(f"Temporary file deleted: {tmp_path}")
    
    return out_path

def samtools_get_cell_reads(bam_path, chr, start, end):
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
        bam_path,
        f"{chr}:{start}-{end}"
    ]
    try:
        if VERBOSE:
            print(f"Executing command: {' '.join(cmd)}")
        region_reads_result = subprocess.run(cmd, stdout=subprocess.PIPE, check = True, text = True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Error message: {e}")
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


def samtools_get_read_count(bam_path, chr, start, end):
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
    # Check if index file exists
    index_path = f"{bam_path}.bai"
    if not os.path.exists(index_path):
        cmd = f"samtools view {bam_path} | wc -l"
        region_reads_result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, check = True, text = True)
    else:
        cmd = [
            "samtools", "view", "-@", str(NCORES),
            "-c", 
            bam_path,
            f"{chr}:{start}-{end}"
        ]
        try:
            if VERBOSE:
                print(f"Executing command: {' '.join(cmd)}")
            region_reads_result = subprocess.run(cmd, stdout=subprocess.PIPE, check = True, text = True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {' '.join(cmd)}")
            print(f"Error message: {e}")
            raise

    read_count = int(region_reads_result.stdout.strip())
    return read_count


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
            print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check = True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Error message: {e}")
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
        if len(group) > 1:
            bam_path = f"{BASEDIR}/data/normal_cell_bams/{group}.bam"
        else:
            bam_path = f"{BASEDIR}/data/normal_cell_bams/group_{group}_merged.bam"
        if not os.path.exists(bam_path):
            raise ValueError(f"BAM file does not exist: {bam_path}")
        group_bam_dict[group] = bam_path

    return group_bam_dict

def get_baseline_bam_old(profile_df, group_bam_dict, profile_name):
    """
    OLD STUFF USE BASH SCRIPT
    Get the baseline bam file containing all reads from baseline cells.
    """
    # Get the baseline cells
    baseline_groups = profile_df[profile_df["clone"] == -1]["cell_group"].str.split(",").explode()
    baseline_cells = profile_df[profile_df["clone"] == -1]["cell_barcode"].str.split(",").explode()
    all_chromosomes = profile_df.loc[1:, "chr"].unique()
    print(all_chromosomes)

    baseline_dict = {}
    for group, barcode in zip(baseline_groups, baseline_cells):
        baseline_dict.setdefault(group, set()).add(barcode)

    # Create a bam file for the baseline cells
    combined_bam_path = f"{OUTDIR}/{profile_name}_baseline_cells.bam"
    with pysam.AlignmentFile(combined_bam_path, "wb", 
                             template = pysam.AlignmentFile(list(group_bam_dict.values())[0], "rb")) as out_bam:
        for group, cells in baseline_dict.items():
            if group not in group_bam_dict:
                raise ValueError(f"BAM file for group {group} does not exist.")

            print(f"Processing BAM file for baseline cells in group {group}...")

            with pysam.AlignmentFile(group_bam_dict[group], "rb") as in_bam:
                for chrom in all_chromosomes:
                    try:
                        for read in in_bam.fetch(str(chrom)):
                            if read.qname in cells:
                                out_bam.write(read)
                    except ValueError:
                        print(f"Chromosome {chrom} not found in BAM file for group {group}. Skipping...")
    print(f"Combined BAM file for baseline cells created at: {combined_bam_path}")


def assign_new_barcodes(sampled_cells_reads_dict, baseline_cell_barcodes):
    
    return


def replace_cell_barcodes(bam_path, new_cb_list, output_bam_path):
    """
    Replace cell barcodes in the BAM file with new cell barcodes.
    """
    bamfile = pysam.AlignmentFile(bam_path, "rb")
    out_bam = pysam.AlignmentFile(output_bam_path, "wb", template=bamfile)

    for i, read in enumerate(bamfile):
        if i >= len(new_cb_list):
            print(f"Warning: Not enough new cell barcodes provided. Stopping at read {i}.")
            raise ValueError("Not enough new cell barcodes provided.")
        read.set_tag("CB", new_cb_list[i])
        out_bam.write(read)
    bamfile.close()
    out_bam.close()

    if len(new_cb_list) > i:
        print(f"Warning: {len(new_cb_list) - i} new cell barcodes were not used.")
    return output_bam_path


def remove_reads_from_baseline_bam(baseline_bam_path, profile_row, profile_row_index, TMP_BAMDIR):
    cell_reads_dict, _ = samtools_get_cell_reads(baseline_bam_path, profile_row.chr,
                                              profile_row.start, profile_row.end)

    # For each cell, select the reads to keep
    all_keep_reads = []
    for cell_barcode, read_ids in cell_reads_dict.items():
        if profile_row.copy_number == -2:
            # Remove all reads from this region in the baseline bam file
            pass
        elif profile_row.copy_number == -1:
            # Remove half reads from this region in the baseline bam file
            n_reads_to_remove = len(read_ids) // 2
            reads_to_keep = np.random.choice(read_ids, size=n_reads_to_remove, replace=False)
            if VERBOSE:
                print(f"Removing {n_reads_to_remove} reads from cell {cell_barcode} in region {profile_row.chr}:{profile_row.start}-{profile_row.end}")
        elif profile_row.copy_number == 0:
            # Keep all reads from this region in the baseline bam file
            reads_to_keep = read_ids
        else:
            raise ValueError(f"Invalid copy number: {profile_row.copy_number}")
        all_keep_reads.extend(reads_to_keep)

    # Save temporary bam file with reads to keep
    out_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_tmp.bam"
    samtools_read_id_filter(baseline_bam_path, out_bam_path, all_keep_reads)

    if VERBOSE:
        print(f"Temporary BAM file created: {out_bam_path}")
    return


def add_reads_to_baseline_bam(baseline_bam_path, baseline_cell_barcodes, group_bam_dict, 
                              profile_row, profile_row_index, TMP_BAMDIR):
    # Check that copy number is greater than 0
    if profile_row.copy_number <= 0:
        raise ValueError(f"Copy number must be greater than 0: {profile_row.copy_number}")

    # Get all necessary groups for this region, assign cell barcodes to groups
    groups = profile_row.cell_group.split(",")
    all_cell_barcodes = profile_row.cell_barcode.split(",")
    group_cell_dict = {}
    for group, cell_barcode in zip(groups, all_cell_barcodes):
        if group not in group_bam_dict:
            raise ValueError(f"BAM file for group {group} does not exist.")
        group_cell_dict.setdefault(group, []).append(cell_barcode)
    unique_groups = list(group_cell_dict.keys())
    group_cell_proportions = np.array([len(group_cell_dict[group]) for group in unique_groups], dtype = float)
    group_cell_proportions /= group_cell_proportions.sum()
    if VERBOSE:
        print(f"Group proportions: {group_cell_proportions}")

    # Figure out how many additional reads to sample based on number of reads in baseline bam file in region
    n_baseline_reads = samtools_get_read_count(baseline_bam_path, profile_row.chr,
                                               profile_row.start, profile_row.end)
    print(n_baseline_reads)
    n_additional_reads = (n_baseline_reads // 2) * profile_row.copy_number - n_baseline_reads

    # Calculate the number of additional reads to sample from each group based on proportion of cells from each group
    n_additional_reads_per_group_array = np.random.multinomial(n_additional_reads, group_cell_proportions)
    n_additional_reads_per_group = dict(zip(groups, n_additional_reads_per_group_array))
    
    # Check that the total number of additional reads is equal to the number of reads to sample
    if sum(n_additional_reads_per_group.values()) != n_additional_reads:
        raise ValueError(f"Total number of additional reads does not match: {sum(n_additional_reads_per_group.values())} != {n_additional_reads}")

    if VERBOSE:
        print(f"Region {profile_row.chr}:{profile_row.start}-{profile_row.end} has {n_baseline_reads} reads in baseline bam file")
        print(f"Sampling {n_additional_reads} additional reads")
        print(f"Number of additional reads per group: {n_additional_reads_per_group}")

    # For each group, get the bam file and sample reads from the group bam file
    tmp_group_bam_paths = [] # 1 temporary bam file per group
    sampled_cells_reads_dict = {} # Dictionary to store sampled reads for each cell barcode
    for group, n_reads in n_additional_reads_per_group.items():
        print(f"Sampling {n_reads} reads from group {group}")
        cell_reads_dict, reads_cell_dict = samtools_get_cell_reads(group_bam_dict[group], profile_row.chr,
                                                                   profile_row.start, profile_row.end)
        all_group_reads = list(read for sublist in cell_reads_dict.values() for read in sublist)
        print(len(all_group_reads))
        print(len(set(all_group_reads)))
        if len(all_group_reads) < n_reads:
            raise ValueError(f"Not enough reads in group {group} for region {profile_row.chr}:{profile_row.start}-{profile_row.end}")
        else:
            sampled_reads = np.random.choice(all_group_reads, size=n_reads, replace=False)
            for read in sampled_reads:
                cell_barcode = reads_cell_dict[read]
                sampled_cells_reads_dict.setdefault(cell_barcode, []).append(read)
        if VERBOSE:
            print(f"Sampling {n_reads} reads from group {group} for region {profile_row.chr}:{profile_row.start}-{profile_row.end}")
        sampled_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_{group}_tmp.bam"
        samtools_read_id_filter(group_bam_dict[group], sampled_bam_path, sampled_reads)
        tmp_group_bam_paths.append(sampled_bam_path)
        if VERBOSE:
            print(f"Temporary BAM file created: {sampled_bam_path}")
    
    # Merge the temporary bam files into a single bam file
    merged_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_merged_tmp.bam"
    if len(tmp_group_bam_paths) < 2:
        n_tmp_reads = samtools_get_read_count(tmp_group_bam_paths[0], profile_row.chr,
                                               profile_row.start, profile_row.end)
        if n_tmp_reads != n_additional_reads:
            raise ValueError(f"Number of reads in temporary bam file does not match: {n_tmp_reads} != {n_additional_reads}")
        os.rename(tmp_group_bam_paths[0], merged_bam_path)
    else:
        samtools_merge(tmp_group_bam_paths, merged_bam_path)
        if VERBOSE:
            print(f"Merged BAM file created: {merged_bam_path}")

    # Check that the number of reads in the merged bam file is equal to the number of additional reads
    n_merged_reads = samtools_get_read_count(merged_bam_path, profile_row.chr,
                                             profile_row.start, profile_row.end)
    if n_merged_reads != n_additional_reads:
        raise ValueError(f"Number of reads in merged bam file does not match: {n_merged_reads} != {n_additional_reads}")
    if VERBOSE:
        print(f"Number of reads in merged bam file: {n_merged_reads}")

    # Remove the temporary bam files
    for tmp_bam_path in tmp_group_bam_paths:
        if os.path.exists(tmp_bam_path):
            os.remove(tmp_bam_path)
            if VERBOSE:
                print(f"Temporary BAM file deleted: {tmp_bam_path}")

    # Randomly assign each new read to a baseline cell barcode - not guaranteed equal distribution
    # new_cell_barcodes = list(np.random.choice(baseline_cell_barcodes, size=n_additional_reads, replace=True))
    new_cell_barcodes = assign_new_barcodes(sampled_cells_reads_dict, baseline_cell_barcodes)
    if VERBOSE:
        print(f"{len(new_cell_barcodes)} new cell barcodes assigned to sampled reads")

    # Rename the cell barcodes in the sampled reads to match the baseline bam file
    out_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_tmp.bam"
    replace_cell_barcodes(merged_bam_path, new_cell_barcodes, out_bam_path)
    
    if VERBOSE:
        print(f"Temporary BAM file with renamed cell barcodes created: {out_bam_path}")
    
    # Remove the merged bam file
    os.remove(merged_bam_path)
    if VERBOSE:
        print(f"Merged BAM file deleted: {merged_bam_path}")

    return


def get_read_count_range(profile_df):
    all_normal_cells_df = pd.read_csv(f"{BASEDIR}/data/all_normal_cells.csv")
    baseline_cells = profile_df[profile_df["clone"] == -1]["cell_barcode"].str.split(",").explode()

    baseline_cells_df = all_normal_cells_df[all_normal_cells_df["cell_barcode"].isin(baseline_cells)]
    read_count_range = baseline_cells_df["total_mapped_reads"].min(), baseline_cells_df["total_mapped_reads"].max()
    print(f"Read count range: {read_count_range}")

    return read_count_range



def sample_cnv_reads(profile_name, group_name):

    # TODO: Function to get list of all groups, read in bam files, dictionary of group:bam path
    if group_name != "":
        profile_path = f"{BASEDIR}/data/small_cnv_profiles/{profile_name}_{group_name}_cnv_profile.tsv"
    else:
        profile_path = f"{BASEDIR}/data/small_cnv_profiles/{profile_name}_cnv_profile.tsv"
    profile_df = pd.read_csv(profile_path, sep="\t")
    group_bam_dict = get_group_bam_dict(profile_df)

    # TODO: Function to get list of baseline cells, make bam file containing all reads from baseline cells
        # Save total number of reads in the baseline bam file
        # Save total number of cells in baseline
    baseline_bam_path = f"{OUTDIR}/{profile_name}_baseline_cells.bam"
    if not os.path.exists(baseline_bam_path):
        ValueError(f"Baseline BAM file does not exist: {baseline_bam_path}")

    # Get list of all baseline cells
    baseline_cell_barcodes = profile_df[profile_df["clone"] == -1]["cell_barcode"].str.split(",").explode()

    TMP_BAMDIR = f"{OUTDIR}/{profile_name}_intermediate_bams"
    os.makedirs(TMP_BAMDIR, exist_ok=True)
    
    # For every row in the profile df (except for first row, which is baseline cells)
    for index, row in profile_df.iterrows():
        if index == 0:
            continue
        break
        # remove_reads_from_baseline_bam(group_bam_dict, row, index)
        # If copy number is 0, skip this region
        # TODO: Function to remove reads from the baseline bam file
            # If copy number is -1, remove half reads from this region in the baseline bam file
            # If copy number is -2, remove all reads from this region in the baseline bam file
        # TODO: Function to sample reads from group bam file (parameters = row in profile df, number of reads to sample)
            # Else sample reads (# reads in baseline bam file for this region) from each of cells in cell_barcode column
            # Remember to reassign cell barcodes to the sampled reads
                # For each cell in the baseline bam file, reorder the cell barcode order randomly
                # Assign the new cell barcodes to the sampled reads


    # TODO: Function to sample original number of reads from the modified bam file

    return


#%%
def main():
    group_name = "sg0_500cells"
    profile_name = "minitest_c3_8"

    sample_cnv_reads(profile_name, group_name)

    return

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Sample CNV reads from BAM files.")
    # parser.add_argument("--profile_path", type=str, required=True, help="Path to the BAM file.")
    # args = parser.parse_args()

    # profile_path = args.profile_path
    group_name = "sg0_500cells"
    profile_name = "minitest_c3_8"

    main()